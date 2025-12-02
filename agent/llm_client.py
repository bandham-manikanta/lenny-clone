"""
NVIDIA NIM API client for Llama model
Handles streaming and non-streaming completions
"""

import os
from typing import Iterator, Optional, Dict, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class NvidiaLlamaClient:
    def __init__(self):
        self.api_key = os.getenv('NVIDIA_API_KEY')
        self.base_url = os.getenv('NVIDIA_BASE_URL')
        
        if not self.api_key or self.api_key == 'PLACEHOLDER':
            raise ValueError("NVIDIA_API_KEY not set in .env file")
        
        # NVIDIA NIM uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Default model - adjust if your NIM uses different model name
        self.model = "meta/llama-3.1-70b-instruct"
        
        print(f"✅ NVIDIA NIM client initialized")
        print(f"   Base URL: {self.base_url}")
        print(f"   Model: {self.model}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Generate completion from NVIDIA NIM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
        
        Returns:
            Complete response string or iterator of chunks
        """
        
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content
        
        except Exception as e:
            print(f"❌ NVIDIA NIM API error: {e}")
            raise
    
    # def _stream_response(self, response) -> Iterator[str]:
    #     """Stream response chunks"""
    #     for chunk in response:
    #         if chunk.choices[0].delta.content:
    #             yield chunk.choices[0].delta.content
    def _stream_response(self, response) -> Iterator[str]:
        """Stream response chunks"""
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> str | Iterator[str]:
        """
        Chat completion with message history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
        
        Returns:
            Complete response string or iterator of chunks
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content
        
        except Exception as e:
            print(f"❌ NVIDIA NIM API error: {e}")
            raise
    # In agent/llm_client.py, add this method:

    def refine_with_style(self, draft_response: str, style_prompt: str) -> str:
        """Second pass: Rewrite in Lenny's voice"""
        
        refinement_prompt = f"""
    You are a style editor. Rewrite this response to match Lenny Rachitsky's voice.

    ORIGINAL RESPONSE:
    {draft_response}

    STYLE REQUIREMENTS:
    {style_prompt}

    REWRITTEN VERSION (Lenny's voice):
    """
        
        return self.generate(
            prompt=refinement_prompt,
            temperature=0.3,  # Lower temp for style consistency
            max_tokens=1500,
            stream=False
        )



if __name__ == "__main__":
    # Test the client
    client = NvidiaLlamaClient()
    
    print("\n🧪 Testing non-streaming...")
    response = client.generate("What is product-market fit?")
    print(f"Response: {response[:200]}...\n")
    
    print("🧪 Testing streaming...")
    for chunk in client.generate("What is product-market fit?", stream=True):
        print(chunk, end="", flush=True)
    print("\n")
