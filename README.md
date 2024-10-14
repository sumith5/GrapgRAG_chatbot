# GrapgRAG_chatbot : Talk to any website
> The main mvp of this GraphRAG_chatbot is to chat with any website by just using the URL of the website you want to chat.

Add the Url in the following function in app.py,find this function and add the url `extract_scrape_content()`

To run the Chatbot application modify the following variables in the app.py
  1. Add openAI key
  2. Add the Neo4j url either free veriosion or localsystem
  3. Add the Neo4j Credentials
```
os.environ["OPENAI_API_KEY"] = ""
os.environ["NEO4J_URI"] = ""
os.environ["NEO4J_USERNAME"] = ""
os.environ["NEO4J_PASSWORD"] = ""
```

**Just Run the cells in the HackUI notebook it will show the application link.**
