from openai import OpenAI

# ==========================DeepSeek========================================
client = OpenAI(api_key="sk-93nWYhI8SrnXad5m9932CeBdDeDf4233B21d93D217095f22", base_url="http://10.2.8.77:3000/v1")

# Round 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="DeepSeek-R1",
    messages=messages,
    stream=True
)

reasoning_content = ""
content = ""

for chunk in response:
    content += chunk.choices[0].delta.content
    print(chunk.choices[0].delta.content,end='')


# ==========================GLM========================================
client = OpenAI(api_key="118aea86606f4e2f82750c54d3bb380c.DxtxnaKQhFz5EHPY", base_url="https://open.bigmodel.cn/api/paas/v4/")

# Round 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="glm-4-flash",
    messages=messages,
    stream=True
)

reasoning_content = ""
content = ""

for chunk in response:
    content += chunk.choices[0].delta.content
    print(chunk.choices[0].delta.content,end='')


