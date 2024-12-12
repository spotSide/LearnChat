import json

class Retrieval:
    def __init__(self, data_path: str = "./data/newjeans.json"):
        try:
            with open(data_path, "r", encoding="utf-8") as file:
                # JSON 파일을 읽어와서 딕셔너리로 저장
                data: dict[str, str] = json.load(file)
                self.data =data
        except FileNotFoundError:
            # 파일이 존재하지 않을 경우 빈 딕셔너리로 초기화
            print(f"파일을 찾을 수 없습니다: {data_path}")
            self.data = {}

    def retrieve(self, query: str) -> str | None:
        # 쿼리를 소문자로 변환하여 대소문자 구분 없이 검색
        query_lower = query.lower()
        for key in self.data.keys():
            # 키를 소문자로 변환하여 쿼리에 포함되어 있는지 확인
            if key.lower() in query_lower:
                # 일치하는 키가 있으면 "키: 값" 형태의 문자열 반환
                return f"{key}: {self.data[key]}"
        # 일치하는 키가 없으면 None 반환
        return None

    def print_data(self) -> None:
        # 데이터 딕셔너리를 예쁘게 출력
        print(json.dumps(self.data, indent=2, ensure_ascii=False))
