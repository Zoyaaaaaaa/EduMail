
# **EduMail: Intelligent Email Processing for EdTech Platforms** 📧🤖

EduMail is an advanced, AI-powered email processing system tailored for EdTech platforms. By leveraging **LangChain**, **LangGraph**, and state-of-the-art NLP models, EduMail automates email handling, categorization, and response generation, making communication efficient and professional.

---

## **🚀 Key Features**

### **1. Automated Email Categorization** 🗂️  
Classifies emails into categories such as:
- **Course Inquiry**
- **Platform Issue**
- **Feedback**
- **Payment Issue**
- **Off-Topic**

This ensures emails are routed for appropriate action.

### **2. Context-Aware Response Generation** 💬  
Generates personalized, professional replies using context extracted from emails and research.

### **3. Dynamic Workflow with LangGraph** 🔄  
EduMail’s workflows are dynamically managed with **LangGraph**, ensuring modular and scalable processing pipelines.

### **4. Research Integration** 🔍  
Fetches additional context using web search APIs for complex queries requiring detailed answers.

### **5. Scalable & Efficient Architecture** 🏗️  
Handles increasing email volumes seamlessly with a highly structured processing pipeline.

---

## **📊 Workflow Overview**

### **Flowchart**  
![image](https://github.com/user-attachments/assets/5bff0b55-bbb8-4f9e-8f1d-6d9c93b887b6)

---

## **🛠️ Technologies Used**

1. **LangChain** 🧠  
   Handles prompt creation, chaining tasks, and ensuring output parsing consistency.

2. **LangGraph** 📈  
   - Manages the conditional logic of the email processing pipeline.  
   - Nodes and edges define distinct tasks like categorization, drafting, analysis, and rewriting.

3. **GROQ LLM** 🤖  
   Powers email categorization, keyword extraction, draft creation, and analysis.

4. **Python** 🐍  
   The core programming language for system implementation and workflows.

5. **Tavily Search** 🔍  
   Fetches relevant information from the web to enhance email responses.

6. **Markdown for Outputs** 📄  
   Saves email drafts and final outputs in markdown format for easy access.

---

## **📂 Key File: `email_agent.py`**

### **Function Highlights**

1. **Categorize Email**  
   Uses AI to classify emails into predefined categories.  
   ```python
   def categorize_email(state):
       email_category = email_category_generator.invoke({"initial_email": initial_email})
   ```

2. **Research Keywords & Web Search**  
   Extracts keywords for context and fetches additional information when necessary.  
   ```python
   keywords = search_keyword_chain.invoke({"initial_email": initial_email, "email_category": email_category})
   ```

3. **Draft Email**  
   Generates the first draft based on the categorized email and available research.  
   ```python
   draft_email = draft_writer_chain.invoke({"initial_email": initial_email, "email_category": email_category})
   ```

4. **Analyze & Rewrite**  
   Provides feedback on drafts and rewrites as needed for professionalism and clarity.  
   ```python
   final_email = rewrite_chain.invoke({...})
   ```

5. **Dynamic Workflow with LangGraph**  
   Manages the full email handling lifecycle through conditional edges and stateful nodes.  
   ```python
   workflow.add_conditional_edges("categorize_email", route_to_research, {...})
   ```

---

## **✨ Benefits**

- **Time-Saving Automation**: Respond to emails quickly and accurately.  
- **Scalable & Modular**: Easily handle increasing email volumes.  
- **High Accuracy**: AI ensures responses are contextually appropriate and professional.  

---

**EduMail** is your AI-powered partner in streamlining EdTech communication, ensuring seamless and intelligent email management! 🌟
