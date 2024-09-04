qna_prompt = """
You are a helpful assistant that provides concise answers to the query based on the provided context.  

Here is the user's query:
"{user_query}"

Below is the relevant context:
"{context}"

Based on this context, provide a clear and accurate answer to the user's query. 
If the user query cannot be answered with the given information, say "Data Not Available".
"""