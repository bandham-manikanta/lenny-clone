from agent.llm_client import NvidiaLlamaClient

client = NvidiaLlamaClient()

print("\n🧪 Test 1: Non-streaming...")
response = client.generate("What is 2+2?", stream=False)
print(f"Response: {response}\n")

print("🧪 Test 2: Streaming...")
chunk_count = 0
for chunk in client.generate("What is 2+2?", stream=True):
    chunk_count += 1
    print(f"Chunk {chunk_count}: {repr(chunk)}")

print(f"\n✅ Total chunks: {chunk_count}")
