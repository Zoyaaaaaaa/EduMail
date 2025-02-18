import os
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants for email
DEFAULT_RECEIVER = os.environ.get("DEFAULT_RECEIVER")
DEFAULT_PASSWORD = os.environ.get("DEFAULT_PASSWORD")

# Define the configuration for agents
config_list = [
    {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "api_key": os.environ.get("TOGETHER_API_KEY"),
        "api_base": "https://api.together.xyz",
        "api_type": "together",
        "stream": False,
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.7,
    "timeout": 120,
}

# Load the reference data
try:
    with open("data.txt", "r") as file:
        reference_data = file.read()
except FileNotFoundError:
    reference_data = ""
    st.error("Error: data.txt not found. Make sure the file is present in the directory.")

def send_email(sender_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = DEFAULT_RECEIVER
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, DEFAULT_PASSWORD)
        text = msg.as_string()
        server.sendmail(sender_email, DEFAULT_RECEIVER, text)
        server.quit()
        return True, "Email sent successfully! ✉️"
    except Exception as e:
        return False, f"Error sending email: {str(e)}"

def initialize_agents():
    categorize_agent = AssistantAgent(
        name="CategorizeAgent",
        llm_config=llm_config,
        system_message="You are an assistant who categorizes emails into 'Sales', 'Custom Enquiry', 'Off Topic', or 'Customer Complaint'. Respond only with the category name.",
    )

    draft_agent = AssistantAgent(
        name="DraftAgent",
        llm_config=llm_config,
        system_message=f"You are an assistant who drafts email responses strictly referencing the following data:\n\n{reference_data}\n\nRespond to all emails professionally and concisely, ensuring accuracy and alignment with the provided data.",
    )

    return categorize_agent, draft_agent

class EmailWorkflow:
    def __init__(self, categorize_agent, draft_agent):
        self.categorize_agent = categorize_agent
        self.draft_agent = draft_agent

    def _get_single_response(self, agent, message):
        agent.reset()
        user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
            llm_config=llm_config,
        )
        user_proxy.reset()
        user_proxy.initiate_chat(agent, message=message)
        return user_proxy.last_message()["content"].strip()

    def process_email(self, message):
        try:
            with st.status("Processing email...", expanded=True) as status:
                status.write("🔍 Categorizing email...")
                category = self._get_single_response(
                    self.categorize_agent,
                    f"Categorize this email:\nMessage: {message}\n\nRespond only with one of these categories: Sales, Custom Enquiry, Off Topic, or Customer Complaint."
                )
                
                status.write("✍️ Drafting response...")
                draft = self._get_single_response(
                    self.draft_agent,
                    f"Draft a professional email reply strictly referencing the provided data:\n\nOriginal Email: {message}\n\nCategory: {category}\n\nEnsure accuracy and alignment with the data. Include a relevant subject line in the format 'Subject: <subject>'"
                )

                subject = "RE: Your Email"
                if "Subject:" in draft:
                    subject_line = [line for line in draft.split('\n') if 'Subject:' in line][0]
                    subject = subject_line.replace('Subject:', '').strip()
                    draft = '\n'.join([line for line in draft.split('\n') if 'Subject:' not in line])
                
                status.write("✅ Processing complete!")
                status.update(state="complete")

            return {
                "category": category,
                "subject": subject,
                "response": draft.strip()
            }

        except Exception as e:
            return {
                "error": f"Error processing email: {str(e)}"
            }

def display_workflow_diagram():
    st.markdown("""
    ### 📊 System Workflow
    """)
    
    st.markdown("""
    ```mermaid
    graph TD
        A[Input Email] --> B[Categorize Agent]
        B --> C{Email Category}
        C -->|Sales| D[Draft Agent]
        C -->|Custom Enquiry| D
        C -->|Off Topic| D
        C -->|Customer Complaint| D
        D --> E[Generate Response]
        E --> F[Review & Edit]
        F --> G[Send Email]
        
    ```
    """)

def main():
    st.set_page_config(
        page_title="AI Email Assistant",
        page_icon="✉️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            margin-top: 10px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .email-container {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #dee2e6;
        }
        .stTextArea textarea {
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        .stTextInput input {
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar with workflow diagram
    with st.sidebar:
        st.title("ℹ️ How it Works")
        display_workflow_diagram()
        
        st.markdown("""
        ### 🔑 Key Features
        - **Smart Categorization**: Automatically identifies email type
        - **Contextual Responses**: Generates replies based on reference data
        - **Professional Format**: Maintains consistent business communication
        - **Easy Review**: Edit before sending
        """)

    st.title("✉️ AI Email Assistant")
    st.markdown("#### Transform your email workflow with AI-powered responses")

    # Initialize session state for agents
    if 'workflow' not in st.session_state:
        categorize_agent, draft_agent = initialize_agents()
        st.session_state.workflow = EmailWorkflow(categorize_agent, draft_agent)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📥 Input Email")
        with st.container(border=True):
            email_message = st.text_area(
                "Enter the email to process:",
                height=200,
                placeholder="Paste your email here..."
            )
            process_btn = st.button("🔄 Process Email", use_container_width=True)
            
            if process_btn and email_message:
                result = st.session_state.workflow.process_email(email_message)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.current_result = result

    with col2:
        st.markdown("### 📤 Send Email")
        with st.container(border=True):
            sender_email = st.text_input("Your Gmail Address:", placeholder="your.email@gmail.com")
            
            if 'current_result' in st.session_state:
                result = st.session_state.current_result
                st.markdown(f"**Category**: {result.get('category', '')}")
                subject = st.text_input("Subject:", value=result.get("subject", ""))
                body = st.text_area("Response:", value=result.get("response", ""), height=200)
                
                send_btn = st.button("📨 Send Email", use_container_width=True)
                if send_btn:
                    if sender_email and subject and body:
                        success, message = send_email(sender_email, subject, body)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.warning("Please fill in all fields before sending.")

    # Display recipient info
    st.markdown("---")
    st.info(f"📌 All emails will be sent to: {DEFAULT_RECEIVER}")

if __name__ == "__main__":
    main()