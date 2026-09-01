When using the application, it will ask the user, "Enter the text you would like to translate: " and "Enter the languages you would like to translate to: ", allowing for multiple languages at the same time.It will then return the text translated into all of the languages, with the name of each language next to its translation. 

It will then return the text translated into all of the languages, with the name of each language next to its translation. If a language is not recognised, a message will be shown saying it is not recognised.

The translation is done by using a Claude API call to translate the text. This code does that:
```python
client = anthropic.Anthropic()
tranlated_text = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    messages=[{"role": "user", "content": f"Can you translate this {text} in these languages? {languages} I want your reply to be structured with the language that it is the translation of, then a colon (:), then have the translated text. Do not include anything apart from this. If a language is not recognised, then replace the translated text with a message saying it is not recognised, and separate each language text pair with a semi-colon (;)."}
    ])
```
This uses an API key that is initialised in the terminal.

The response will come back with a lot of other unwanted information around it. Using `translated_text.content[0].text` will strip everything out except the response itself.