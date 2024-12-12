from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import gradio as gr

class Generator:
    def __init__(self, model_path: str) -> None:
        """
        Args:
            model_path (str): Hugging Face 모델 저장소의 모델 이름 또는 로컬 모델 경로.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # GPU를 사용하는 경우
            # torch_dtype=torch.float32,  # CPU를 사용하는 경우
            device_map="auto",
            trust_remote_code=True
        )
        self.streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    def generate(self, prompt: str) -> str:
        """
        주어진 프롬프트에 이어지는 자연스러운 텍스트를 생성합니다.

        Args:
            prompt (str): 텍스트 생성의 시작점이 되는 프롬프트.

        Returns:
            str: 생성된 텍스트.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        # 텍스트 생성 (스트리밍 출력을 사용하지 않음)
        generation_output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # 생성된 텍스트 디코딩
        generated_text = self.tokenizer.decode(
            generation_output[0], skip_special_tokens=True
        )

        # 프롬프트를 제외한 생성된 부분만 반환
        return generated_text[len(prompt):]

    def generate_response(self, prompt, history):
        """
        Gradio ChatInterface에서 사용할 챗봇 응답 생성 함수
        """
        # 이전 대화 기록과 새로운 입력 결합
        messages = []
        for user_message, bot_message in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_message})
        messages.append({"role": "user", "content": prompt})

        # 입력 인코딩
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(self.model.device)
    
        # 텍스트 생성 (스트리밍 사용)
        generation_output = self.model.generate(
            input_ids=input_ids,
            streamer=self.streamer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        return ""

# 모델 경로 설정 (필요에 따라 수정)
model_path = "/home/intel/LearnChat/lama/phi-3-mini"

# Generator 인스턴스 생성
generator = Generator(model_path)

# Gradio 인터페이스 생성
iface = gr.ChatInterface(
    fn=generator.generate_response,  # Generator 인스턴스의 generate_response 사용
    chatbot=gr.Chatbot(height=400, type="messages"),
    textbox=gr.Textbox(placeholder="Type your message here...", container=False, scale=7),
    title="Phi-3-mini Chatbot",
    description="This is a simple chatbot powered by Phi-3-mini model.",
    theme="soft",
    examples=["What is the capital of France?", "Tell me a joke.", "How to make a cake?"],
    cache_examples=False,
)

# 인터페이스 실행
iface.launch(share=True)
