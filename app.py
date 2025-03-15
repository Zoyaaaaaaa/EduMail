import os
import streamlit as st
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration and styling
st.set_page_config(
    page_title="AI Email Assistant",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #0D47A1;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-message {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #43A047;
    }
    .error-message {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #E53935;
    }
    .stTextArea > div > div > textarea {
        border-radius: 5px;
        border: 1px solid #BBDEFB;
    }
</style>
""", unsafe_allow_html=True)

# Define the configuration for agents
def get_llm_config():
    config_list = [
        {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "api_key": os.environ.get("TOGETHER_API_KEY"),
            "api_base": "https://api.together.xyz",
            "api_type": "together",
            "stream": False,
        }
    ]
    
    return {
        "config_list": config_list,
        "temperature": 0.7,
        "timeout": 120,
    }

def send_email(sender_email, receiver_email, subject, body, sender_password):
    """Send an email using Gmail SMTP with improved error handling and security."""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Create secure connection using SSL context
        context = ssl.create_default_context()
        
        # Connect to Gmail's SMTP server
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls(context=context)
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            
        return True, "Email sent successfully! ✉️"
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Please check your email and password. If using Gmail, make sure to use an App Password."
    except smtplib.SMTPServerDisconnected:
        return False, "Server disconnected. Check your internet connection."
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except Exception as e:
        return False, f"Error sending email: {str(e)}"

def initialize_agents(reference_data):
    """Initialize the AI agents with the provided reference data."""
    llm_config = get_llm_config()
    
    categorize_agent = AssistantAgent(
        name="CategorizeAgent",
        llm_config=llm_config,
        system_message="""You categorize emails into one of these categories:
        - 'Sales': Inquiries about purchasing products or services
        - 'Customer Enquiry': General questions about products, services, or policies
        - 'Customer Complaint': Issues, problems, or dissatisfaction with products or services
        - 'Off Topic': Messages unrelated to business operations
        
        Respond only with the category name.""",
    )
    
    draft_agent = AssistantAgent(
        name="DraftAgent",
        llm_config=llm_config,
        system_message=f"""Draft professional, empathetic email responses using this reference data:

{reference_data}

Your responses should:
1. Address the sender by name if available
2. Begin with a personalized greeting
3. Acknowledge their message
4. Provide relevant information from the reference data
5. Answer their questions or resolve their issues
6. Include a friendly closing
7. Add a professional signature""",
    )
    
    return categorize_agent, draft_agent

class EmailWorkflow:
    """Handles the email processing workflow using multiple agents."""
    def __init__(self, categorize_agent, draft_agent):
        self.categorize_agent = categorize_agent
        self.draft_agent = draft_agent
        self.llm_config = get_llm_config()

    def _get_single_response(self, agent, message):
        """Get a single response from an agent."""
        agent.reset()
        user_proxy = UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
            llm_config=self.llm_config,
        )
        user_proxy.reset()
        user_proxy.initiate_chat(agent, message=message)
        return user_proxy.last_message()["content"].strip()

    def process_email(self, message):
        """Process an email to categorize it and draft a response."""
        try:
            with st.spinner("Categorizing email..."):
                category = self._get_single_response(
                    self.categorize_agent,
                    f"Categorize this email:\nMessage: {message}\nRespond with one category: Sales, Customer Enquiry, Off Topic, or Customer Complaint."
                )
            
            with st.spinner("Drafting response..."):
                draft = self._get_single_response(
                    self.draft_agent,
                    f"Draft a professional response using reference data:\n\nOriginal Email: {message}\nCategory: {category}"
                )
            
            return {"category": category, "response": draft.strip()}
        except Exception as e:
            return {"error": f"Error processing email: {str(e)}"}

def display_sidebar():
    """Display the sidebar with information about the app."""
    with st.sidebar:
        st.markdown("## 📚 About")
        st.markdown("""
        This AI Email Assistant helps you:
        - Process incoming emails
        - Categorize them automatically
        - Generate professional responses
        - Send replies directly
        
        Powered by AutoGen and Mixtral AI.
        """)
        
        st.markdown("## 🔒 Security Note")
        st.markdown("""
        Email credentials are used only for sending emails and are not stored.
        For Gmail users, use an App Password instead of your main password.
        [Learn how to create an App Password](https://support.google.com/accounts/answer/185833)
        """)
        
        st.markdown("## 📝 Categories")
        st.markdown("""
        - **Sales**: Purchase inquiries
        - **Customer Enquiry**: General questions
        - **Customer Complaint**: Issues or problems
        - **Off Topic**: Unrelated messages
        """)

def main():
    display_sidebar()
    
    # Main header
    st.markdown("<h1 class='main-header'>✉️ AI Email Assistant</h1>", unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📚 Knowledge Base", "📥 Process Email", "📤 Send Email"])
    
    with tab1:
        st.markdown("<div class='info-box'>Enter information about your products, services, pricing, policies, and common responses here. This knowledge will be used to generate accurate email replies.</div>", unsafe_allow_html=True)
        
        # Knowledge base input
        if 'reference_data' not in st.session_state:
            st.session_state.reference_data = ""
        
        st.session_state.reference_data = st.text_area(
            "Knowledge Base / Reference Data:",
            st.session_state.reference_data,
            height=300,
            placeholder="Example:\nProduct line: Eco-friendly water bottles ($20-$35)\nReturn policy: 30-day money-back guarantee\nShipping: Free for orders over $50\nContact: support@example.com, phone: (555) 123-4567"
        )
        
        if st.button("💾 Save Knowledge Base"):
            if st.session_state.reference_data.strip():
                with st.spinner("Initializing AI agents..."):
                    categorize_agent, draft_agent = initialize_agents(st.session_state.reference_data)
                    st.session_state.workflow = EmailWorkflow(categorize_agent, draft_agent)
                st.markdown("<div class='success-message'>✅ Knowledge base saved successfully! You can now process emails.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='error-message'>⚠️ Please enter some reference data before proceeding.</div>", unsafe_allow_html=True)
    
    with tab2:
        # Email Processing
        if 'workflow' not in st.session_state:
            st.markdown("<div class='info-box'>⚠️ Please save a knowledge base in the first tab before processing emails.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 class='sub-header'>📥 Process Email</h2>", unsafe_allow_html=True)
            
            email_message = st.text_area(
                "Enter the email to process:", 
                height=200,
                placeholder="Example:\nFrom: customer@example.com\nSubject: Question about water bottle durability\n\nHello,\nI'm interested in your eco-friendly water bottles but wondering how durable they are. Do they have any warranty?\nThanks,\nAlex"
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                process_btn = st.button("🔄 Process Email")
            
            if process_btn and email_message:
                result = st.session_state.workflow.process_email(email_message)
                
                if "error" in result:
                    st.markdown(f"<div class='error-message'>❌ {result['error']}</div>", unsafe_allow_html=True)
                else:
                    st.session_state.current_result = result
                    st.markdown(f"<div class='success-message'>✅ Email processed! Category: {result['category']}</div>", unsafe_allow_html=True)
                    
                    # Show the generated response
                    st.subheader("Generated Response:")
                    st.text_area("", result["response"], height=200)
                    
                    # Switch to the Send Email tab
                    st.success("👉 Go to the 'Send Email' tab to review and send this response")
    
    with tab3:
        # Email Sending
        if 'current_result' not in st.session_state:
            st.markdown("<div class='info-box'>⚠️ Process an email in the second tab before sending a response.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 class='sub-header'>📤 Send Email</h2>", unsafe_allow_html=True)
            
            # Email form with columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                sender_email = st.text_input(
                    "Your Email Address:", 
                    placeholder="your.email@gmail.com"
                )
                sender_password = st.text_input(
                    "Your Email Password:", 
                    type="password", 
                    help="For Gmail users, use an App Password instead of your regular password"
                )
            
            with col2:
                receiver_email = st.text_input(
                    "Recipient Email Address:", 
                    placeholder="customer@example.com"
                )
                subject = st.text_input(
                    "Subject:", 
                    value=f"Re: {st.session_state.current_result.get('category', '')}"
                )
            
            st.subheader("Email Body:")
            body = st.text_area(
                "", 
                value=st.session_state.current_result.get("response", ""), 
                height=300
            )
            
            col1, col2 = st.columns([1, 3])
            with col1:
                send_btn = st.button("📨 Send Email")
            
            if send_btn:
                if sender_email and sender_password and receiver_email and subject and body:
                    with st.spinner("Sending email..."):
                        success, message = send_email(sender_email, receiver_email, subject, body, sender_password)
                    
                    if success:
                        st.markdown(f"<div class='success-message'>✅ {message}</div>", unsafe_allow_html=True)
                        
                        # Create a record of the sent email
                        if 'sent_emails' not in st.session_state:
                            st.session_state.sent_emails = []
                        
                        st.session_state.sent_emails.append({
                            "to": receiver_email,
                            "subject": subject,
                            "category": st.session_state.current_result.get("category", "")
                        })
                        
                        # Clear current result to prevent duplicate sends
                        st.session_state.pop('current_result', None)
                    else:
                        st.markdown(f"<div class='error-message'>❌ {message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='error-message'>⚠️ Please fill in all fields before sending.</div>", unsafe_allow_html=True)
    
    # Display sent emails history at the bottom if any
    if 'sent_emails' in st.session_state and st.session_state.sent_emails:
        st.markdown("<h2 class='sub-header'>📋 Email History</h2>", unsafe_allow_html=True)
        
        email_data = []
        for i, email in enumerate(st.session_state.sent_emails):
            email_data.append([i+1, email["to"], email["subject"], email["category"]])
        
        st.table({
            "#": [row[0] for row in email_data],
            "Recipient": [row[1] for row in email_data],
            "Subject": [row[2] for row in email_data],
            "Category": [row[3] for row in email_data]
        })

if __name__ == "__main__":
    main()
