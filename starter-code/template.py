"""
Lab #3: Baseline Chatbot vs ReAct Agent
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from tools import TOOL_DEFINITIONS, TOOL_MAP


SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.

Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:

Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
...
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""


class ChatbotBaseline:
    """
    Chatbot cơ sở, không sử dụng ReAct Loop hay Tools.
    """

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
        }


class ReActAgent:
    """
    ReAct Agent sử dụng vòng lặp:

    Thought -> Action -> Observation -> Final Answer
    """

    def __init__(
        self,
        max_iterations: int = 5,
        api_key: Optional[str] = None,
    ):
        self.max_iterations = max(1, int(max_iterations))
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        self.trace: List[Dict[str, Any]] = []
        self._plan: List[Dict[str, Any]] = []
        self._observations: List[Any] = []
        self._plan_input: Optional[str] = None

    def parse_city_code(self, text: str) -> str:
        """
        Chuẩn hóa tên thành phố thành mã sân bay.
        """

        text_upper = text.upper()

        for code in ["SGN", "HAN", "DAD"]:
            if code in text_upper:
                return code

        if "HÀ NỘI" in text_upper:
            return "HAN"

        if (
            "HỒ CHÍ MINH" in text_upper
            or "SÀI GÒN" in text_upper
        ):
            return "SGN"

        if "ĐÀ NẴNG" in text_upper:
            return "DAD"

        return "SGN"

    @staticmethod
    def parse_action(
        raw_action: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Phân tích và kiểm tra Action JSON.
        """

        if isinstance(raw_action, str):
            try:
                raw_action = json.loads(raw_action)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON format") from exc

        if not isinstance(raw_action, dict):
            raise ValueError("Action must be a JSON object")

        name = str(raw_action.get("name", "")).strip().lower()
        args = raw_action.get("args", {})

        if not name:
            raise ValueError("Action name is required")

        if not isinstance(args, dict):
            raise ValueError("Action args must be an object")

        return {
            "name": name,
            "args": args,
        }

    @staticmethod
    def _extract_max_price(text: str) -> int:
        """
        Đọc giá tối đa từ các dạng:
        - dưới 2 triệu
        - dưới 1.5 triệu
        - tối đa 2000000
        """

        pattern = (
            r"(?:dưới|<=|tối đa|không quá)"
            r"\s*([\d.,]+)\s*(triệu|tr|m)?"
        )

        match = re.search(pattern, text.lower())

        if not match:
            return 5_000_000

        number_text = match.group(1).replace(",", "")
        unit = (match.group(2) or "").lower()

        try:
            number = (
                float(number_text)
                if "." in number_text
                else int(number_text)
            )
        except ValueError:
            return 5_000_000

        if unit in {"triệu", "tr", "m"}:
            number *= 1_000_000

        return int(number)

    def _extract_route(self, text: str) -> Tuple[str, str]:
        """
        Lấy mã sân bay điểm đi và điểm đến.
        """

        codes = re.findall(
            r"\b(?:SGN|HAN|DAD)\b",
            text.upper(),
        )

        if len(codes) >= 2:
            return codes[0], codes[1]

        city_names = [
            "HÀ NỘI",
            "HỒ CHÍ MINH",
            "SÀI GÒN",
            "ĐÀ NẴNG",
        ]

        city_codes = []

        for city in city_names:
            if city in text.upper():
                city_codes.append(
                    self.parse_city_code(city)
                )

        if len(city_codes) >= 2:
            return city_codes[0], city_codes[1]

        return "HAN", "SGN"

    def _build_plan(
        self,
        user_input: str,
    ) -> List[Dict[str, Any]]:
        """
        Xây dựng kế hoạch hành động dựa trên câu hỏi.
        """

        lower_text = user_input.lower()
        upper_text = user_input.upper()

        plan: List[Dict[str, Any]] = []

        code_count = len(
            re.findall(
                r"\b(?:SGN|HAN|DAD)\b",
                upper_text,
            )
        )

        city_names = [
            "HÀ NỘI",
            "HỒ CHÍ MINH",
            "SÀI GÒN",
            "ĐÀ NẴNG",
        ]

        city_count = sum(
            city in upper_text
            for city in city_names
        )

        has_route = code_count >= 2 or city_count >= 2

        # Tránh nhầm câu hỏi “chính sách đổi trả vé”
        # thành yêu cầu tìm chuyến bay.
        asks_flight = (
            "chuyến bay" in lower_text
            or (
                "vé" in lower_text
                and has_route
            )
        )

        asks_weather = any(
            phrase in lower_text
            for phrase in [
                "thời tiết",
                "nhiệt độ",
                "mặc gì",
                "trang phục",
            ]
        )

        if asks_flight:
            origin, destination = self._extract_route(
                user_input
            )

            plan.append(
                {
                    "name": "get_flight_info",
                    "args": {
                        "origin": origin,
                        "destination": destination,
                        "max_price": self._extract_max_price(
                            user_input
                        ),
                    },
                }
            )

        if asks_weather:
            plan.append(
                {
                    "name": "get_weather_forecast",
                    "args": {
                        "city_code": self.parse_city_code(
                            user_input
                        )
                    },
                }
            )

        return plan

    def _execute_action(
        self,
        action: Dict[str, Any],
    ) -> Any:
        """
        Tìm và thực thi tool trong TOOL_MAP.
        """

        try:
            action = self.parse_action(action)

            tool_name = action["name"].strip().lower()
            tool = TOOL_MAP.get(tool_name)

            if tool is None:
                return {
                    "error": f"Unknown tool: {tool_name}"
                }

            return tool(**action["args"])

        except Exception as exc:
            return {
                "error": str(exc)
            }

    @staticmethod
    def _observation_text(
        observation: Any,
    ) -> str:
        if isinstance(observation, (dict, list)):
            return json.dumps(
                observation,
                ensure_ascii=False,
            )

        return str(observation)

    def _compose_answer(
        self,
        user_input: str,
    ) -> str:
        """
        Tổng hợp các Observation thành Final Answer.
        """

        if not self._plan:
            return (
                "Vinpearl thường áp dụng chính sách "
                "đổi/trả theo điều kiện của từng loại vé. "
                "Vui lòng kiểm tra điều kiện vé hoặc liên hệ "
                "bộ phận hỗ trợ."
            )

        parts: List[str] = []

        for action, observation in zip(
            self._plan,
            self._observations,
        ):
            if action["name"] == "get_flight_info":
                if isinstance(observation, list) and observation:
                    flights = []

                    for flight in observation:
                        flights.append(
                            f"{flight.get('flight_number')} "
                            f"({flight.get('airline')}, "
                            f"{flight.get('price_vnd'):,} VND, "
                            f"{flight.get('departure_time')})"
                        )

                    parts.append(
                        "Chuyến bay phù hợp: "
                        + "; ".join(flights)
                        + "."
                    )
                else:
                    parts.append(
                        "Không tìm thấy chuyến bay phù hợp."
                    )

            elif action["name"] == "get_weather_forecast":
                if (
                    isinstance(observation, dict)
                    and "error" not in observation
                ):
                    parts.append(
                        f"Thời tiết tại "
                        f"{observation.get('city')}: "
                        f"{observation.get('temperature_c')}°C, "
                        f"{observation.get('condition')}. "
                        f"Gợi ý: "
                        f"{observation.get('recommendation')}"
                    )
                else:
                    error_message = (
                        observation.get(
                            "error",
                            "Không xác định",
                        )
                        if isinstance(observation, dict)
                        else "Không xác định"
                    )

                    parts.append(
                        "Không lấy được thông tin thời tiết: "
                        f"{error_message}."
                    )

        return " ".join(parts)

    def plan_and_execute_step(
        self,
        user_input: str,
        iteration: int,
    ) -> Tuple[str, bool]:
        """
        Thực thi một bước ReAct.

        Trả về:
        - Nội dung Observation hoặc Final Answer
        - True nếu agent đã hoàn tất
        """

        if self._plan_input != user_input:
            self._plan_input = user_input
            self._plan = self._build_plan(user_input)
            self._observations = []

        # Trường hợp câu hỏi không cần tool.
        if not self._plan:
            answer = self._compose_answer(user_input)

            self.trace.append(
                {
                    "step": iteration,
                    "thought": "Không cần gọi tool.",
                    "action": None,
                    "observation": answer,
                }
            )

            return answer, True

        # Thực thi tool tương ứng với iteration hiện tại.
        if iteration <= len(self._plan):
            action = self._plan[iteration - 1]
            observation = self._execute_action(action)

            self._observations.append(observation)

            self.trace.append(
                {
                    "step": iteration,
                    "thought": (
                        f"Cần dùng {action['name']} "
                        "để lấy dữ liệu."
                    ),
                    "action": action,
                    "observation": observation,
                }
            )

            # Nếu chỉ có một tool thì hoàn tất ngay.
            # Nếu có nhiều tool thì cần thêm bước tổng hợp.
            is_done = len(self._plan) == 1

            return (
                self._observation_text(observation),
                is_done,
            )

        # Bước tổng hợp Final Answer.
        answer = self._compose_answer(user_input)

        self.trace.append(
            {
                "step": iteration,
                "thought": (
                    "Đã đủ dữ liệu, tạo Final Answer."
                ),
                "action": None,
                "observation": answer,
            }
        )

        return answer, True

    def run(
        self,
        user_input: str,
    ) -> Dict[str, Any]:
        """
        Chạy ReAct Loop cho đến khi hoàn tất
        hoặc đạt giới hạn max_iterations.
        """

        self.trace = []
        self._plan = self._build_plan(user_input)
        self._plan_input = user_input
        self._observations = []

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            _, done = self.plan_and_execute_step(
                user_input,
                iteration,
            )

            if done:
                return {
                    "status": "completed",
                    "iterations": len(self.trace),
                    "answer": self._compose_answer(
                        user_input
                    ),
                    "trace": self.trace,
                }

        return {
            "status": "max_iterations_reached",
            "iterations": len(self.trace),
            "answer": (
                "Không thể hoàn thành trong số bước tối đa."
            ),
            "trace": self.trace,
        }


def main() -> None:
    user_query = (
        "Tìm cho tôi chuyến bay từ HAN đi SGN "
        "dưới 2 triệu, rồi cho biết thời tiết SGN "
        "nên mặc gì?"
    )

    print("=== RUNNING CHATBOT BASELINE ===")

    chatbot = ChatbotBaseline()

    print(
        json.dumps(
            chatbot.query(user_query),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== RUNNING REACT AGENT ===")

    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()