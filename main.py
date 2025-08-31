import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import FastAPI, HTTPException, Request, Form, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
import uvicorn

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
import hashlib
import uuid
import logging
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Email Assistant",
    description="Scalable AI-powered email processing and response system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
except Exception:
    templates = None
    logger.warning("Templates directory not found")

# Pydantic models
class CompanyKnowledgeBase(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=100)
    knowledge_data: str = Field(..., min_length=10)
    contact_email: EmailStr
    
class EmailProcessRequest(BaseModel):
    company_name: str = Field(..., min_length=1)
    email_content: str = Field(..., min_length=10)
    sender_email: Optional[EmailStr] = None
    
class EmailSendRequest(BaseModel):
    sender_email: EmailStr
    sender_password: str = Field(..., min_length=1)
    receiver_email: EmailStr
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    company_name: str

class EmailResponse(BaseModel):
    category: str
    response: str
    confidence_score: float
    processing_time: float

class VectorDBManager:
    """Manages Qdrant vector database operations with connection handling"""
    
    def __init__(self, qdrant_url: str = None, api_key: str = None):
        # Use cloud Qdrant credentials
        self.qdrant_url = qdrant_url or os.environ.get(
            "QDRANT_URL", 
            "https://fd4fbfdd-437f-4a3a-b960-ba26c0f15635.eu-west-2-0.aws.cloud.qdrant.io:6333"
        )
        self.api_key = api_key or os.environ.get(
            "QDRANT_API_KEY", 
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.MqWlmyumL1de56Hdf5qFYWL6IS8Wo8vbWXsbKmYT55s"
        )
        self.client = None
        self.embedding_model = None
        self.vector_size = 384
        self.is_connected = False
        self._initialize()
        
    def _initialize(self):
        """Initialize embedding model and connect to Qdrant"""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
            
            # Connect to Qdrant
            self._connect_with_retry()
        except Exception as e:
            logger.error(f"Error initializing VectorDBManager: {str(e)}")
        
    def _connect_with_retry(self, max_retries: int = 3, retry_delay: int = 2):
        """Connect to Qdrant with retry logic"""
        for attempt in range(max_retries):
            try:
                self.client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.api_key,
                )
                # Test connection
                collections = self.client.get_collections()
                self.is_connected = True
                logger.info(f"Successfully connected to Qdrant cloud")
                logger.info(f"Found {len(collections.collections)} existing collections")
                return
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} to connect to Qdrant failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to Qdrant after {max_retries} attempts")
                    self.is_connected = False
    
    def _ensure_connection(self) -> bool:
        """Ensure Qdrant connection is active"""
        if not self.is_connected or self.client is None:
            logger.warning("Qdrant not connected. Attempting to reconnect...")
            self._connect_with_retry()
        return self.is_connected
    
    def create_collection(self, company_name: str) -> bool:
        """Create a collection for a company if it doesn't exist"""
        if not self._ensure_connection():
            logger.error("Cannot create collection: Qdrant not connected")
            return False
            
        try:
            collection_name = self._get_collection_name(company_name)
            
            # Check if collection exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name not in existing_names:
                # Create collection with vector config
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                
                # Create index for company_name field to enable filtering
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name="company_name",
                        field_schema="keyword"
                    )
                    logger.info(f"Created collection with index: {collection_name}")
                except Exception as index_error:
                    logger.warning(f"Collection created but index creation failed: {str(index_error)}")
            else:
                logger.info(f"Collection {collection_name} already exists")
            
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            self.is_connected = False
            return False
    
    def add_knowledge_base(self, company_name: str, knowledge_data: str, metadata: Dict[str, Any]) -> bool:
        """Add knowledge base data to company collection"""
        if not self._ensure_connection():
            logger.error("Cannot add knowledge base: Qdrant not connected")
            return False
            
        try:
            collection_name = self._get_collection_name(company_name)
            
            # Create collection if it doesn't exist
            if not self.create_collection(company_name):
                return False
            
            # Split knowledge data into chunks for better retrieval
            chunks = self._split_text(knowledge_data)
            points = []
            
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_model.encode(chunk).tolist()
                point_id = str(uuid.uuid4())
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "company_name": company_name,
                        "chunk_index": i,
                        "created_at": datetime.utcnow().isoformat(),
                        **metadata
                    }
                )
                points.append(point)
            
            # Upsert points to collection
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Added {len(points)} knowledge chunks for {company_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge base: {str(e)}")
            self.is_connected = False
            return False
    
    def search_knowledge(self, company_name: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant knowledge in company collection"""
        if not self._ensure_connection():
            logger.error("Cannot search knowledge: Qdrant not connected")
            return []
            
        try:
            collection_name = self._get_collection_name(company_name)
            
            # Check if collection exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name not in existing_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            # Encode query
            query_vector = self.embedding_model.encode(query).tolist()
            
            # Search in collection without problematic filter
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit * 2,
                score_threshold=0.3,
            )
            
            # Filter results by company_name in the payload
            filtered_results = []
            for hit in search_result:
                if hit.payload.get("company_name") == company_name:
                    filtered_results.append({
                        "text": hit.payload["text"],
                        "score": hit.score,
                        "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                    })
                if len(filtered_results) >= limit:
                    break
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {str(e)}")
            return []
    
    def _get_collection_name(self, company_name: str) -> str:
        """Generate collection name from company name"""
        safe_name = "".join(c.lower() if c.isalnum() else "_" for c in company_name)
        return f"company_{safe_name}"
    
    def _split_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split text into chunks for better vector storage"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) + 1 > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [text]

class EmailAgentManager:
    """Manages AI agents for email processing with improved context handling"""
    
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.llm_config = self._get_llm_config()
        
    def _get_llm_config(self):
        """Get LLM configuration for Gemini API with validation"""
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            return None
        
        # Validate API key format (basic check)
        if len(gemini_api_key) < 20:
            logger.warning("GEMINI_API_KEY appears to be invalid (too short)")
            return None
        
        config_list = [
            {
                "model": "gemini-2.5-flash",
                "api_key": gemini_api_key,
                "api_type": "google",
            }
        ]
        
        return {
            "config_list": config_list,
            "temperature": 0.7,
            "timeout": 120,
        }
    
    def categorize_email(self, email_content: str) -> str:
        """Categorize email with improved logic"""
        try:
            # If no valid LLM config, use basic categorization
            if not self.llm_config:
                logger.info("Using basic categorization due to missing/invalid API key")
                return self._basic_categorize(email_content)
            
            # Try AI categorization with Gemini
            categorize_agent = AssistantAgent(
                name="CategorizeAgent",
                llm_config=self.llm_config,
                system_message="""You categorize emails into one of these categories:
                - 'Sales': Inquiries about purchasing products or services
                - 'Customer Enquiry': General questions about products, services, or policies
                - 'Customer Complaint': Issues, problems, or dissatisfaction with products or services
                - 'Off Topic': Messages unrelated to business operations
                
                Respond only with the category name.""",
            )
            
            user_proxy = UserProxyAgent(
                name="UserProxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
                llm_config=self.llm_config,
            )
            
            user_proxy.initiate_chat(
                categorize_agent, 
                message=f"Categorize this email:\n{email_content}\nRespond with one category: Sales, Customer Enquiry, Off Topic, or Customer Complaint."
            )
            
            category = user_proxy.last_message()["content"].strip()
            
            # Validate category response
            valid_categories = ["Sales", "Customer Enquiry", "Customer Complaint", "Off Topic"]
            if category not in valid_categories:
                logger.warning(f"Invalid category returned: {category}, using basic categorization")
                return self._basic_categorize(email_content)
            
            return category
            
        except Exception as e:
            logger.error(f"Error categorizing email with Gemini: {str(e)}")
            return self._basic_categorize(email_content)
    
    def _basic_categorize(self, email_content: str) -> str:
        """Enhanced keyword-based categorization fallback"""
        content_lower = email_content.lower()
        
        # Sales keywords
        sales_keywords = ["buy", "purchase", "price", "cost", "quote", "order", "sale", "pricing", "plan", "package"]
        if any(keyword in content_lower for keyword in sales_keywords):
            return "Sales"
        
        # Complaint keywords
        complaint_keywords = ["problem", "issue", "complain", "wrong", "error", "bad", "terrible", "not working", "broken", "help"]
        if any(keyword in content_lower for keyword in complaint_keywords):
            return "Customer Complaint"
        
        # Off topic keywords
        off_topic_keywords = ["weather", "news", "politics", "random", "spam"]
        if any(keyword in content_lower for keyword in off_topic_keywords):
            return "Off Topic"
        
        # Default to Customer Enquiry
        return "Customer Enquiry"
    
    def draft_response(self, company_name: str, email_content: str, category: str) -> str:
        """Draft email response using company knowledge base with improved context handling"""
        try:
            # Search for relevant knowledge first
            relevant_knowledge = self.vector_db.search_knowledge(company_name, email_content, limit=5)
            
            # If we have good knowledge and valid LLM config, use AI
            if relevant_knowledge and self.llm_config:
                return self._ai_draft_response(company_name, email_content, category, relevant_knowledge)
            
            # If we have knowledge but no AI, use context-aware template
            elif relevant_knowledge:
                return self._context_aware_response(company_name, email_content, category, relevant_knowledge)
            
            # Otherwise use basic response
            else:
                logger.info(f"No relevant knowledge found for {company_name}, using basic response")
                return self._basic_response(company_name, email_content, category)
                
        except Exception as e:
            logger.error(f"Error drafting response: {str(e)}")
            return self._basic_response(company_name, email_content, category)
    
    def _ai_draft_response(self, company_name: str, email_content: str, category: str, relevant_knowledge: List[Dict]) -> str:
        """Use AI to draft response with context using Gemini"""
        try:
            # Combine knowledge into context
            knowledge_context = "\n".join([item["text"] for item in relevant_knowledge])
            print("KNOWLEDGE CONTEXT",knowledge_context)
            draft_agent = AssistantAgent(
                name="DraftAgent",
                llm_config=self.llm_config,
                system_message=f"""You are an expert customer service representative for {company_name}. 

Use this company-specific knowledge to draft your response:
{knowledge_context}

Guidelines:
1. Address the customer professionally and warmly
2. Acknowledge their specific concern or question
3. Provide relevant information from the company knowledge above
4. Give specific, actionable solutions when possible
5. Include appropriate contact information if mentioned in knowledge
6. Close with a professional signature

Category: {category}
Make your response specific to their actual question, not generic.""",
            )
            
            user_proxy = UserProxyAgent(
                name="UserProxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
                llm_config=self.llm_config,
            )
            
            user_proxy.initiate_chat(
                draft_agent,
                message=f"Draft a specific, helpful response for this {category} email:\n\n{email_content}\n\nUse the company knowledge provided to give specific solutions and information."
            )
            
            response = user_proxy.last_message()["content"].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error in AI draft with Gemini: {str(e)}")
            # Fall back to context-aware response
            return self._context_aware_response(company_name, email_content, category, relevant_knowledge)
    
    def _context_aware_response(self, company_name: str, email_content: str, category: str, relevant_knowledge: List[Dict]) -> str:
        """Generate context-aware response using retrieved knowledge without AI"""
        try:
            # Extract relevant information from knowledge
            knowledge_text = " ".join([item["text"] for item in relevant_knowledge])
            
            # Create response based on category and context
            if category == "Customer Complaint":
                if "login" in email_content.lower() and "login" in knowledge_text.lower():
                    return self._create_login_help_response(company_name, knowledge_text)
                elif "billing" in email_content.lower() and "billing" in knowledge_text.lower():
                    return self._create_billing_help_response(company_name, knowledge_text)
                elif "password" in email_content.lower() and "password" in knowledge_text.lower():
                    return self._create_password_help_response(company_name, knowledge_text)
                else:
                    return self._create_generic_complaint_response(company_name, knowledge_text)
            
            elif category == "Sales":
                return self._create_sales_response(company_name, knowledge_text)
            
            elif category == "Customer Enquiry":
                return self._create_enquiry_response(company_name, knowledge_text)
            
            else:  # Off Topic
                return self._basic_response(company_name, email_content, category)
                
        except Exception as e:
            logger.error(f"Error creating context-aware response: {str(e)}")
            return self._basic_response(company_name, email_content, category)
    
    def _create_login_help_response(self, company_name: str, knowledge_text: str) -> str:
        """Create specific login help response"""
        # Extract relevant login information
        response = f"""Dear Customer,

Thank you for contacting {company_name}. I understand you're experiencing difficulties with the login process, and I'm here to help resolve this issue quickly.

Based on our support documentation, here are the steps to resolve login issues:

"""
        
        # Add specific steps from knowledge if available
        if "forgot password" in knowledge_text.lower():
            response += "1. If you've forgotten your password, please use the 'Forgot Password' link on the login page\n"
        if "reset link" in knowledge_text.lower():
            response += "2. Check your email for a password reset link\n"
        if "strong password" in knowledge_text.lower():
            response += "3. Create a new strong password with numbers and symbols\n"
        if "24 hours" in knowledge_text:
            response += "4. Please note that reset links expire in 24 hours\n"
        if "support@" in knowledge_text:
            support_email = "support@platform.com"  # Extract from knowledge if needed
            response += f"5. If you continue to have issues, contact our support team at {support_email}\n"
        
        response += f"""
If you need immediate assistance or these steps don't resolve your issue, please don't hesitate to reach out to our technical support team.

We apologize for any inconvenience and appreciate your patience.

Best regards,
{company_name} Customer Support Team"""
        
        return response
    
    def _create_billing_help_response(self, company_name: str, knowledge_text: str) -> str:
        """Create specific billing help response"""
        response = f"""Dear Valued Customer,

Thank you for reaching out to {company_name} regarding your billing concerns. We take all billing matters seriously and are here to help resolve any issues.

To assist you with your billing inquiry, please try the following steps:

"""
        
        # Add specific billing steps from knowledge
        if "payment method" in knowledge_text.lower():
            response += "1. Verify your payment method details in your account settings\n"
        if "expired cards" in knowledge_text.lower():
            response += "2. Check if your payment card has expired and update if necessary\n"
        if "billing@" in knowledge_text:
            response += "3. For billing disputes or detailed inquiries, contact billing@platform.com\n"
        if "grace period" in knowledge_text.lower():
            response += "4. Please note we provide a 3-day grace period for payment issues\n"
        
        response += f"""
Our billing team will review your account and contact you within 24 hours to resolve any outstanding issues.

We value your business and apologize for any inconvenience this may have caused.

Best regards,
{company_name} Billing Support Team"""
        
        return response
    
    def _create_password_help_response(self, company_name: str, knowledge_text: str) -> str:
        """Create specific password help response"""
        response = f"""Dear Customer,

Thank you for contacting {company_name}. I understand you need assistance with password-related issues.

Here's how to resolve password problems:

"""
        
        if "password reset page" in knowledge_text.lower():
            response += "1. Visit our password reset page\n"
        if "registered email" in knowledge_text.lower():
            response += "2. Enter your registered email address\n"
        if "reset email" in knowledge_text.lower():
            response += "3. Follow the instructions in the reset email we send you\n"
        if "security questions" in knowledge_text.lower():
            response += "4. You may need to answer security questions for verification\n"
        if "two-factor authentication" in knowledge_text.lower():
            response += "5. We recommend enabling two-factor authentication for added security\n"
        
        response += f"""
If you continue to experience issues, our support team is available to assist you further.

Best regards,
{company_name} Technical Support Team"""
        
        return response
    
    def _create_sales_response(self, company_name: str, knowledge_text: str) -> str:
        """Create sales response using available knowledge"""
        response = f"""Dear Prospective Customer,

Thank you for your interest in {company_name}! We're excited to help you find the right solution for your needs.

"""
        
        # Add relevant information from knowledge
        if "pricing" in knowledge_text.lower():
            response += "Based on your inquiry, I'd be happy to provide you with detailed pricing information and available packages.\n\n"
        
        if "subscription" in knowledge_text.lower():
            response += "We offer various subscription plans to meet different needs and budgets.\n\n"
        
        response += f"""Our sales team will contact you within 24 hours to discuss your specific requirements and provide personalized recommendations.

If you have any immediate questions, please feel free to reach out to us directly.

Looking forward to serving you!

Best regards,
{company_name} Sales Team"""
        
        return response
    
    def _create_enquiry_response(self, company_name: str, knowledge_text: str) -> str:
        """Create enquiry response using available knowledge"""
        response = f"""Dear Customer,

Thank you for contacting {company_name}. We appreciate your inquiry and are here to provide you with the information you need.

"""
        
        # Add context-specific information
        if len(knowledge_text) > 100:
            response += "Based on our available resources, we'll provide you with comprehensive information to address your questions.\n\n"
        
        response += f"""Our customer service team is reviewing your message and will provide you with detailed information within 24 hours.

If you need immediate assistance, please don't hesitate to contact us directly.

Best regards,
{company_name} Customer Service Team"""
        
        return response
    
    def _create_generic_complaint_response(self, company_name: str, knowledge_text: str) -> str:
        """Create generic complaint response with available context"""
        response = f"""Dear Valued Customer,

Thank you for bringing your concerns to our attention. At {company_name}, we take all customer feedback seriously and are committed to resolving any issues you may experience.

"""
        
        if "support@" in knowledge_text:
            response += "Our dedicated support team will investigate your concerns and work towards a swift resolution.\n\n"
        
        response += f"""We sincerely apologize for any inconvenience this may have caused. Our customer service team will investigate your concerns immediately and contact you within 24 hours with a resolution plan.

Your satisfaction is our priority, and we appreciate your patience as we work to resolve this matter.

Best regards,
{company_name} Customer Service Team"""
        
        return response
    
    def _basic_response(self, company_name: str, email_content: str, category: str) -> str:
        """Basic response template when no context is available"""
        responses = {
            "Sales": f"""Dear Valued Customer,

Thank you for your interest in {company_name}. We appreciate you reaching out to us regarding our products and services.

We have received your inquiry and one of our sales representatives will contact you within 24 hours to discuss your requirements and provide you with detailed information.

If you have any urgent questions, please don't hesitate to contact us directly.

Best regards,
{company_name} Customer Service Team""",
            
            "Customer Enquiry": f"""Dear Customer,

Thank you for contacting {company_name}. We have received your inquiry and appreciate you taking the time to reach out to us.

Our customer service team is reviewing your message and will provide you with a comprehensive response within 24 hours. We are committed to addressing all your questions and concerns.

If you need immediate assistance, please feel free to contact us directly.

Best regards,
{company_name} Customer Service Team""",
            
            "Customer Complaint": f"""Dear Valued Customer,

Thank you for bringing your concerns to our attention. At {company_name}, we take all customer feedback seriously and are committed to resolving any issues you may have experienced.

We sincerely apologize for any inconvenience this may have caused. Our customer service team will investigate your concerns immediately and contact you within 24 hours with a resolution plan.

Your satisfaction is our priority, and we appreciate your patience as we work to resolve this matter.

Best regards,
{company_name} Customer Service Team""",
            
            "Off Topic": f"""Dear Sender,

Thank you for your message. While we appreciate you reaching out to {company_name}, your inquiry appears to be outside the scope of our business operations.

If you have any questions related to our products or services, please feel free to contact us again.

Best regards,
{company_name} Customer Service Team"""
        }
        
        return responses.get(category, responses["Customer Enquiry"])

# Additional utility function to test Gemini API key
def test_gemini_api_key() -> bool:
    """Test if Gemini API key is valid"""
    try:
        import requests
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return False
        
        # Test API call with a simple request
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        
        response = requests.get(url, timeout=10)
        return response.status_code == 200
        
    except Exception as e:
        logger.error(f"Error testing Gemini API key: {str(e)}")
        return False

class EmailSender:
    """Handles email sending functionality"""
    
    @staticmethod
    def send_email(sender_email: str, receiver_email: str, subject: str, body: str, sender_password: str) -> tuple[bool, str]:
        """Send an email using Gmail SMTP"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Create secure connection
            context = ssl.create_default_context()
            
            # Connect to Gmail SMTP server
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls(context=context)
                server.login(sender_email, sender_password)
                text = msg.as_string()
                server.sendmail(sender_email, receiver_email, text)
                
            return True, "Email sent successfully!"
            
        except smtplib.SMTPAuthenticationError:
            return False, "Authentication failed. Please check your email and password."
        except smtplib.SMTPServerDisconnected:
            return False, "Server disconnected. Check your internet connection."
        except Exception as e:
            return False, f"Error sending email: {str(e)}"

# Initialize global components
vector_db = VectorDBManager()
agent_manager = EmailAgentManager(vector_db)
email_sender = EmailSender()

# Routes
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting AI Email Assistant application with Gemini API")
    logger.info(f"Qdrant connection status: {'Connected' if vector_db.is_connected else 'Disconnected'}")
    
    # Test Gemini API key on startup
    gemini_status = test_gemini_api_key()
    logger.info(f"Gemini API status: {'Valid' if gemini_status else 'Invalid or not configured'}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse("""
        <html>
            <head><title>AI Email Assistant</title></head>
            <body>
                <h1>AI Email Assistant API</h1>
                <p>Welcome to the AI Email Assistant API powered by Google Gemini. Use the following endpoints:</p>
                <ul>
                    <li>POST /api/knowledge-base - Add company knowledge</li>
                    <li>POST /api/process-email - Process emails</li>
                    <li>POST /api/send-email - Send emails</li>
                    <li>GET /api/companies - List companies</li>
                    <li>GET /health - Health check</li>
                    <li>GET /docs - API documentation</li>
                </ul>
            </body>
        </html>
        """)

@app.post("/api/knowledge-base")
async def add_knowledge_base(knowledge: CompanyKnowledgeBase):
    """Add or update company knowledge base"""
    try:
        # Check if Qdrant is connected
        if not vector_db.is_connected:
            raise HTTPException(
                status_code=503, 
                detail="Vector database is not available. Please try again later."
            )
        
        metadata = {
            "contact_email": knowledge.contact_email,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        success = vector_db.add_knowledge_base(
            company_name=knowledge.company_name,
            knowledge_data=knowledge.knowledge_data,
            metadata=metadata
        )
        
        if success:
            return {
                "status": "success", 
                "message": f"Knowledge base for {knowledge.company_name} updated successfully",
                "company_name": knowledge.company_name,
                "chunks_added": len(vector_db._split_text(knowledge.knowledge_data))
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update knowledge base")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in add_knowledge_base: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/process-email", response_model=EmailResponse)
async def process_email(request: EmailProcessRequest):
    """Process an email and generate a response using Gemini"""
    try:
        start_time = datetime.utcnow()
        
        # Test API key before processing
        api_key_valid = test_gemini_api_key()
        if not api_key_valid:
            logger.warning("Gemini API key is invalid, using fallback methods")
        
        # Categorize email
        category = agent_manager.categorize_email(request.email_content)
        logger.info(f"Email categorized as: {category}")
        
        # Draft response
        response = agent_manager.draft_response(
            company_name=request.company_name,
            email_content=request.email_content,
            category=category
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Calculate confidence score based on knowledge retrieval
        confidence_score = 0.5  # Default confidence
        if vector_db.is_connected:
            relevant_knowledge = vector_db.search_knowledge(request.company_name, request.email_content, limit=1)
            if relevant_knowledge:
                confidence_score = min(relevant_knowledge[0]["score"], 0.95)  # Cap at 95%
            else:
                confidence_score = 0.3  # Lower confidence when no knowledge found
        
        return EmailResponse(
            category=category,
            response=response,
            confidence_score=confidence_score,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing email: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/send-email")
async def send_email_endpoint(email_request: EmailSendRequest, background_tasks: BackgroundTasks):
    """Send an email"""
    try:
        success, message = email_sender.send_email(
            sender_email=email_request.sender_email,
            receiver_email=email_request.receiver_email,
            subject=email_request.subject,
            body=email_request.body,
            sender_password=email_request.sender_password
        )
        
        if success:
            logger.info(f"Email sent from {email_request.sender_email} to {email_request.receiver_email}")
            return {
                "status": "success", 
                "message": message,
                "sent_at": datetime.utcnow().isoformat(),
                "from": email_request.sender_email,
                "to": email_request.receiver_email,
                "subject": email_request.subject
            }
        else:
            return {"status": "error", "message": message}
            
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/companies")
async def get_companies():
    """Get list of companies with knowledge bases"""
    try:
        if not vector_db.is_connected:
            return {"companies": [], "warning": "Vector database not connected"}
            
        collections = vector_db.client.get_collections()
        companies = []
        
        for collection in collections.collections:
            if collection.name.startswith("company_"):
                try:
                    # Get collection info for vector count
                    collection_info = vector_db.client.get_collection(collection.name)
                    company_name = collection.name.replace("company_", "").replace("_", " ").title()
                    
                    # Get vector count from collection info
                    vector_count = 0
                    if hasattr(collection_info, 'vectors_count'):
                        vector_count = collection_info.vectors_count
                    elif hasattr(collection_info, 'points_count'):
                        vector_count = collection_info.points_count
                    
                    companies.append({
                        "name": company_name,
                        "collection": collection.name,
                        "vectors_count": vector_count,
                        "created_at": datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for collection {collection.name}: {str(e)}")
                    company_name = collection.name.replace("company_", "").replace("_", " ").title()
                    companies.append({
                        "name": company_name,
                        "collection": collection.name,
                        "vectors_count": 0,
                        "error": "Could not retrieve collection details"
                    })
        
        return {
            "companies": companies,
            "total_companies": len(companies),
            "qdrant_status": "connected"
        }
        
    except Exception as e:
        logger.error(f"Error getting companies: {str(e)}")
        return {"companies": [], "error": str(e), "qdrant_status": "error"}

@app.get("/api/companies/{company_name}/knowledge")
async def get_company_knowledge(company_name: str, limit: int = 10):
    """Get knowledge base entries for a specific company"""
    try:
        if not vector_db.is_connected:
            raise HTTPException(status_code=503, detail="Vector database not connected")
        
        collection_name = vector_db._get_collection_name(company_name)
        
        # Get points from collection
        points = vector_db.client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )[0]  # scroll returns (points, next_page_offset)
        
        knowledge_entries = []
        for point in points:
            knowledge_entries.append({
                "id": point.id,
                "text": point.payload.get("text", ""),
                "metadata": {k: v for k, v in point.payload.items() if k not in ["text", "company_name"]}
            })
        
        return {
            "company_name": company_name,
            "knowledge_entries": knowledge_entries,
            "total_entries": len(knowledge_entries)
        }
        
    except Exception as e:
        logger.error(f"Error getting company knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/companies/{company_name}")
async def delete_company(company_name: str):
    """Delete a company's knowledge base"""
    try:
        if not vector_db.is_connected:
            raise HTTPException(status_code=503, detail="Vector database not connected")
        
        collection_name = vector_db._get_collection_name(company_name)
        
        # Delete collection
        vector_db.client.delete_collection(collection_name)
        
        return {
            "status": "success",
            "message": f"Knowledge base for {company_name} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting company: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    qdrant_status = "connected" if vector_db.is_connected else "disconnected"
    embedding_status = "loaded" if vector_db.embedding_model else "not_loaded"
    gemini_status = "configured" if os.environ.get("GEMINI_API_KEY") else "not_configured"
    
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "qdrant": qdrant_status,
            "embedding_model": embedding_status,
            "gemini_api": gemini_status
        },
        "version": "1.0.0"
    }

@app.get("/api/test-connection")
async def test_connections():
    """Test all service connections"""
    results = {}
    
    # Test Qdrant connection
    try:
        if vector_db.client:
            collections = vector_db.client.get_collections()
            results["qdrant"] = {
                "status": "connected",
                "collections_count": len(collections.collections)
            }
        else:
            results["qdrant"] = {"status": "not_connected"}
    except Exception as e:
        results["qdrant"] = {"status": "error", "message": str(e)}
    
    # Test embedding model
    try:
        if vector_db.embedding_model:
            test_embedding = vector_db.embedding_model.encode("test")
            results["embedding"] = {
                "status": "loaded",
                "dimension": len(test_embedding)
            }
        else:
            results["embedding"] = {"status": "not_loaded"}
    except Exception as e:
        results["embedding"] = {"status": "error", "message": str(e)}
    
    # Test Gemini API configuration
    try:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if gemini_api_key:
            api_valid = test_gemini_api_key()
            results["gemini_api"] = {
                "status": "valid" if api_valid else "invalid",
                "configured": True
            }
        else:
            results["gemini_api"] = {
                "status": "not_configured", 
                "message": "GEMINI_API_KEY not found",
                "configured": False
            }
    except Exception as e:
        results["gemini_api"] = {"status": "error", "message": str(e)}
    
    return {
        "overall_status": "healthy" if all(r.get("status") in ["connected", "loaded", "valid"] for r in results.values()) else "degraded",
        "services": results,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/search-test/{company_name}")
async def test_search(company_name: str, query: str = "test query"):
    """Test search functionality for a specific company"""
    try:
        if not vector_db.is_connected:
            raise HTTPException(status_code=503, detail="Vector database not connected")
        
        results = vector_db.search_knowledge(company_name, query, limit=3)
        
        return {
            "company_name": company_name,
            "query": query,
            "results_count": len(results),
            "results": results,
            "search_successful": len(results) > 0
        }
        
    except Exception as e:
        logger.error(f"Error in search test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-email-processing")
async def test_email_processing():
    """Test email processing with sample data"""
    try:
        # Sample email for testing
        sample_email = {
            "company_name": "TestCompany",
            "email_content": "Hi, I'm interested in purchasing your premium package. Can you provide pricing information and availability?",
            "sender_email": "test@example.com"
        }
        
        # Process the sample email
        request = EmailProcessRequest(**sample_email)
        result = await process_email(request)
        
        return {
            "test_status": "success",
            "sample_input": sample_email,
            "result": result.dict(),
            "message": "Email processing test completed successfully with Gemini API"
        }
        
    except Exception as e:
        logger.error(f"Error in email processing test: {str(e)}")
        return {
            "test_status": "error",
            "message": str(e)
        }

@app.get("/api/gemini-status")
async def gemini_status():
    """Check Gemini API status and configuration"""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            return {
                "status": "not_configured",
                "message": "GEMINI_API_KEY environment variable not set",
                "configured": False
            }
        
        # Test the API key
        is_valid = test_gemini_api_key()
        
        return {
            "status": "valid" if is_valid else "invalid",
            "configured": True,
            "api_key_length": len(api_key),
            "message": "Gemini API is working" if is_valid else "Gemini API key is invalid or API is unreachable",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking Gemini status: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "configured": False
        }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )