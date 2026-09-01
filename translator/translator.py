import anthropic

text=input("enter the text you would like to translate: ")
languages=input("enter the languages you would like to translate to: ")

client = anthropic.Anthropic()
translated_text = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    messages=[{"role": "user", "content": f"Can you translate this {text} in these languages? {languages} I want your reply to be structured with the language that it is the translation of, then a colon (:), then have the translated text. Do not include anything apart from this. If a language is not recognised, then replace the translated text with a message saying it is not recognised, and separate each language text pair with a semi-colon (;)."}
    ])

translated_text_neat = translated_text.content[0].text
translated_texts=translated_text_neat.split(";")
for i in translated_texts:
    print(i.strip())