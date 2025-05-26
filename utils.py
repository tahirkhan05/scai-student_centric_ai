import streamlit as st
import os
import requests
import json
import time
import random
from datetime import datetime, timedelta
import base64
import io
from PIL import Image
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import PyPDF2
import docx
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@st.cache_resource
def setup_ai():
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("Please set GOOGLE_API_KEY in your .env file")
            return None
        
        genai.configure(api_key=api_key)
        
        # Test the API key
        model = genai.GenerativeModel('gemini-1.5-flash')
        test_response = model.generate_content("Hello")
        
        return model
    except Exception as e:
        st.error(f"AI setup failed: {str(e)}")
        return None

def search_courses_online(topic, difficulty="All", min_rating=0):
    """Search for courses online using Gemini AI"""
    model = setup_ai()
    if not model:
        return []
    
    try:
        prompt = f"""
        Find 10 online courses about "{topic}" with difficulty level "{difficulty}" and minimum rating {min_rating}.
        Return the results in this exact JSON format:
        {{
            "courses": [
                {{
                    "title": "Course Title",
                    "provider": "Platform Name",
                    "instructor": "Instructor/Author Name",
                    "difficulty": "Beginner/Intermediate/Advanced",
                    "rating": 4.5,
                    "url": "https://example.com",
                    "description": "Brief description"
                }}
            ]
        }}
        
        Make sure to include real course titles, providers like Coursera, edX, Udemy, Khan Academy, and instructor names.
        """
        
        response = model.generate_content(prompt)
        
        # Try to parse JSON from response
        text = response.text
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0]
        else:
            json_str = text
        
        try:
            data = json.loads(json_str)
            courses = data.get('courses', [])
            
            # Filter by difficulty and rating
            filtered_courses = []
            for course in courses:
                if difficulty != "All" and course.get('difficulty', '').lower() != difficulty.lower():
                    continue
                if course.get('rating', 0) < min_rating:
                    continue
                filtered_courses.append(course)
            
            return filtered_courses
        except json.JSONDecodeError:
            # Fallback: generate sample courses
            return generate_sample_courses(topic, difficulty)
            
    except Exception as e:
        st.error(f"Course search failed: {str(e)}")
        return generate_sample_courses(topic, difficulty)

def generate_sample_courses(topic, difficulty):
    """Generate sample courses as fallback"""
    providers = ["Coursera", "edX", "Udemy", "Khan Academy", "MIT OpenCourseWare"]
    instructors = ["Dr. Smith", "Prof. Johnson", "Sarah Wilson", "Michael Chen", "Dr. Brown"]
    courses = []
    
    for i in range(5):
        course = {
            "title": f"{topic} - {['Fundamentals', 'Advanced', 'Practical', 'Complete Guide', 'Masterclass'][i]}",
            "provider": random.choice(providers),
            "instructor": random.choice(instructors),
            "difficulty": difficulty if difficulty != "All" else random.choice(["Beginner", "Intermediate", "Advanced"]),
            "rating": round(random.uniform(4.0, 5.0), 1),
            "url": f"https://example.com/course-{i+1}",
            "description": f"Comprehensive course covering {topic} concepts and practical applications."
        }
        courses.append(course)
    
    return courses

def generate_learning_path(courses, topic, background="", user_difficulty="Beginner"):
    """Generate learning path using AI with user difficulty consideration"""
    model = setup_ai()
    if not model or not courses:
        return "Learning path could not be generated."
    
    try:
        # Sort courses by difficulty level for better progression
        difficulty_order = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
        sorted_courses = sorted(courses, key=lambda x: difficulty_order.get(x.get('difficulty', 'Beginner'), 1))
        
        course_list = "\n".join([f"- {c['title']} ({c['difficulty']}) - {c['provider']} by {c.get('instructor', 'N/A')}" for c in sorted_courses[:5]])
        background_text = f"Student background: {background}\n" if background else ""
        
        # Customize prompt based on user's difficulty level
        difficulty_guidance = {
            "Beginner": "Focus on foundational concepts, provide extra explanations for basic terms, and suggest more beginner-friendly resources first.",
            "Intermediate": "Build upon existing knowledge, introduce intermediate concepts progressively, and balance theory with practical applications.",
            "Advanced": "Emphasize advanced topics, assume solid foundational knowledge, and focus on specialized skills and cutting-edge developments."
        }
        
        guidance = difficulty_guidance.get(user_difficulty, difficulty_guidance["Beginner"])
        
        prompt = f"""
        Create a detailed learning path for "{topic}" tailored to a {user_difficulty} level learner using these courses:
        {course_list}
        
        {background_text}
        User's current level: {user_difficulty}
        
        Guidance for this level: {guidance}
        
        Structure the path with:
        1. Prerequisites and foundational knowledge (adjust based on {user_difficulty} level)
        2. Step-by-step progression appropriate for {user_difficulty} learners
        3. Estimated timeline for each phase (realistic for {user_difficulty} level)
        4. Key skills to develop at each stage
        5. Practical projects matching {user_difficulty} complexity
        6. Recommended course sequence starting with most appropriate difficulty level
        
        Make it comprehensive, actionable, and specifically tailored to {user_difficulty} level learners.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating learning path: {str(e)}"

def generate_educational_hook(topic, style, tone, output_size):
    """Generate educational hooks using AI"""
    model = setup_ai()
    if not model:
        return "AI model not available"
    
    try:
        size_map = {
            "Small": "in 2-3 sentences",
            "Medium": "in 1-2 paragraphs", 
            "Large": "in 3-4 detailed paragraphs"
        }
        
        size_instruction = size_map.get(output_size, "in 1-2 paragraphs")
        
        prompt = f"""
        Create an engaging educational hook about "{topic}" with the following specifications:
        - Style: {style}
        - Tone: {tone}
        - Length: {size_instruction}
        
        Make it captivating and designed to grab student attention immediately.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating hook: {str(e)}"

def process_uploaded_document(uploaded_file):
    """Process uploaded document and extract text with support for multiple formats"""
    st.session_state.total_documents_processed += 1

    try:
        file_content = ""
        file_type = uploaded_file.type.lower()
        
        # Plain text files
        if file_type == "text/plain":
            file_content = str(uploaded_file.read(), "utf-8")
            
        # PDF files
        elif file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        file_content += f"\n--- Page {page_num + 1} ---\n{text}\n"
                except Exception as e:
                    file_content += f"\n--- Page {page_num + 1} (Error reading) ---\n"
            
            if not file_content.strip():
                file_content = "PDF content could not be extracted. The file may be image-based or corrupted."
                
        # DOCX files
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(BytesIO(uploaded_file.read()))
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
            file_content = "\n".join(paragraphs)
            
        # Legacy DOC files
        elif file_type in ["application/msword", "application/vnd.ms-word"]:
            file_content = "Legacy DOC format detected. Please convert to DOCX or PDF for better text extraction."
            
        # CSV files
        elif file_type == "text/csv" or uploaded_file.name.endswith('.csv'):
            csv_content = str(uploaded_file.read(), "utf-8")
            file_content = f"CSV Data:\n{csv_content}"
            
        # JSON files
        elif file_type == "application/json" or uploaded_file.name.endswith('.json'):
            json_content = str(uploaded_file.read(), "utf-8")
            try:
                # Pretty print JSON for better readability
                import json
                parsed_json = json.loads(json_content)
                file_content = f"JSON Data:\n{json.dumps(parsed_json, indent=2)}"
            except json.JSONDecodeError:
                file_content = f"JSON Data (raw):\n{json_content}"
                
        # XML files
        elif file_type in ["application/xml", "text/xml"] or uploaded_file.name.endswith('.xml'):
            xml_content = str(uploaded_file.read(), "utf-8")
            file_content = f"XML Data:\n{xml_content}"
            
        # Markdown files
        elif file_type == "text/markdown" or uploaded_file.name.endswith(('.md', '.markdown')):
            md_content = str(uploaded_file.read(), "utf-8")
            file_content = f"Markdown Content:\n{md_content}"
            
        # RTF files
        elif file_type == "application/rtf" or uploaded_file.name.endswith('.rtf'):
            rtf_content = str(uploaded_file.read(), "utf-8")
            # Basic RTF parsing - remove RTF control codes for better readability
            import re
            # Remove RTF control words and brackets
            cleaned = re.sub(r'\\[a-z]+\d*\s?', '', rtf_content)
            cleaned = re.sub(r'[{}]', '', cleaned)
            file_content = f"RTF Content:\n{cleaned}"
            
        # HTML files
        elif file_type == "text/html" or uploaded_file.name.endswith(('.html', '.htm')):
            html_content = str(uploaded_file.read(), "utf-8")
            # Basic HTML tag removal for better text extraction
            import re
            # Remove HTML tags
            cleaned = re.sub(r'<[^>]+>', '', html_content)
            # Decode HTML entities
            import html
            cleaned = html.unescape(cleaned)
            file_content = f"HTML Content:\n{cleaned}"
            
        # Code files (various programming languages)
        elif uploaded_file.name.endswith(('.py', '.js', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs', '.swift')):
            code_content = str(uploaded_file.read(), "utf-8")
            file_extension = uploaded_file.name.split('.')[-1]
            file_content = f"Code File (.{file_extension}):\n{code_content}"
            
        # Configuration files
        elif uploaded_file.name.endswith(('.yml', '.yaml', '.ini', '.cfg', '.conf', '.toml')):
            config_content = str(uploaded_file.read(), "utf-8")
            file_extension = uploaded_file.name.split('.')[-1]
            file_content = f"Configuration File (.{file_extension}):\n{config_content}"
            
        # Log files
        elif uploaded_file.name.endswith(('.log', '.logs')):
            log_content = str(uploaded_file.read(), "utf-8")
            file_content = f"Log File:\n{log_content}"
            
        # LaTeX files
        elif uploaded_file.name.endswith(('.tex', '.latex')):
            latex_content = str(uploaded_file.read(), "utf-8")
            file_content = f"LaTeX Document:\n{latex_content}"
            
        # Default fallback - attempt to read as text
        else:
            try:
                file_content = str(uploaded_file.read(), "utf-8")
                file_content = f"Text Content (Auto-detected):\n{file_content}"
            except UnicodeDecodeError:
                file_content = f"Unsupported file format: {uploaded_file.type}. File appears to be binary or uses unsupported encoding."
        
        # Final validation
        if not file_content.strip():
            return "No readable content found in the uploaded file."
        
        # Truncate if too long (optional - prevents memory issues)
        max_length = 50000  # Adjust as needed
        if len(file_content) > max_length:
            file_content = file_content[:max_length] + "\n\n[Content truncated due to length...]"
            
        return file_content
        
    except Exception as e:
        return f"Error processing document: {str(e)}. Please ensure the file is not corrupted and try again."

def chat_with_document(question, document_content):
    """Enhanced chat with document using RAG approach"""
    st.session_state.total_questions_asked += 1

    model = setup_ai()
    if not model:
        return "AI model not available"
    
    try:
        # Split document into chunks for better processing
        max_chunk_size = 4000
        chunks = []
        
        if len(document_content) > max_chunk_size:
            # Split into overlapping chunks
            chunk_size = max_chunk_size
            overlap = 200
            
            for i in range(0, len(document_content), chunk_size - overlap):
                chunk = document_content[i:i + chunk_size]
                chunks.append(chunk)
        else:
            chunks = [document_content]
        
        # Find most relevant chunks based on question keywords
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            # Simple relevance scoring based on keyword overlap
            overlap_score = len([word for word in question_words if word in chunk_lower])
            chunk_scores.append((i, overlap_score, chunk))
        
        # Sort by relevance and take top chunks
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = chunk_scores[:3]  # Use top 3 most relevant chunks
        
        # Combine relevant content
        relevant_content = "\n\n".join([chunk[2] for chunk in relevant_chunks])
        
        prompt = f"""
        You are an AI assistant analyzing a document. Answer the user's question based ONLY on the provided document content.
        
        Document Content:
        {relevant_content}
        
        User Question: {question}
        
        Instructions:
        1. Provide a detailed and accurate answer based solely on the document content
        2. If the answer is not in the document, clearly state that the information is not available
        3. Quote relevant sections when appropriate
        4. If the question requires information from multiple parts of the document, synthesize them coherently
        5. Be specific and cite page numbers or sections when available
        
        Answer:
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error analyzing document: {str(e)}"

def process_image_with_ai(image, question):
    """Process image with specific question using Gemini Vision"""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return "Google API key not found. Please set GOOGLE_API_KEY in your .env file"
        
        genai.configure(api_key=api_key)
        
        # Use Gemini Pro Vision model for image analysis
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Convert PIL image to bytes for API
        img_byte_arr = io.BytesIO()
        
        # Handle different image formats
        if image.mode == 'RGBA':
            # Convert RGBA to RGB for JPEG compatibility
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1])
            rgb_image.save(img_byte_arr, format='JPEG', quality=95)
        elif image.mode == 'P':
            # Convert palette mode to RGB
            rgb_image = image.convert('RGB')
            rgb_image.save(img_byte_arr, format='JPEG', quality=95)
        else:
            # For RGB and other modes, save as JPEG
            image.save(img_byte_arr, format='JPEG', quality=95)
        
        img_byte_arr.seek(0)
        
        # Create the image part for the API
        image_part = {
            "mime_type": "image/jpeg",
            "data": img_byte_arr.getvalue()
        }
        
        prompt = f"""
        Analyze this image and answer the following question: {question}
        
        Provide detailed analysis based on what you can see in the image. 
        Be specific about colors, objects, text, people, scenes, or any other relevant details.
        If the question asks for educational content, provide explanations suitable for learning.
        """
        
        # Generate content with both text and image
        response = model.generate_content([prompt, image_part])
        return response.text
        
    except Exception as e:
        # Fallback to text-only analysis if image processing fails
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            fallback_prompt = f"""
            I cannot process the uploaded image directly, but I can help answer questions about images in general.
            
            Question: {question}
            
            Please provide a general response about this type of question, and suggest what to look for in images when answering such questions.
            """
            response = model.generate_content(fallback_prompt)
            return f"⚠️ Image processing unavailable. General guidance:\n\n{response.text}"
        except:
            return f"Error processing image: {str(e)}. Please ensure you have a valid Google API key and the image is in a supported format (JPG, PNG, JPEG)."
        
# Generate test questions
def generate_test_questions(topic, difficulty, num_questions):
    """Generate test questions using AI"""
    model = setup_ai()
    if not model:
        return [], []
    
    try:
        prompt = f"""
        Generate {num_questions} multiple choice questions about "{topic}" at {difficulty} level.
        
        Return in this exact JSON format:
        {{
            "questions": [
                {{
                    "question": "What is...?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": 0,
                    "explanation": "Why this is correct..."
                }}
            ]
        }}
        
        Make questions challenging and educational.
        """
        
        response = model.generate_content(prompt)
        text = response.text
        
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0]
        else:
            json_str = text
        
        try:
            data = json.loads(json_str)
            questions = data.get('questions', [])
            return questions, [q.get('correct_answer', 0) for q in questions]
        except:
            # Fallback sample questions
            return generate_sample_questions_fallback(topic, num_questions)
            
    except Exception as e:
        return generate_sample_questions_fallback(topic, num_questions)

def generate_sample_questions_fallback(topic, num_questions):
    """Generate sample questions as fallback"""
    questions = []
    answers = []
    
    for i in range(min(num_questions, 5)):
        question = {
            "question": f"Sample question {i+1} about {topic}?",
            "options": [f"Option A for Q{i+1}", f"Option B for Q{i+1}", f"Option C for Q{i+1}", f"Option D for Q{i+1}"],
            "correct_answer": random.randint(0, 3),
            "explanation": f"This is the explanation for question {i+1}."
        }
        questions.append(question)
        answers.append(question['correct_answer'])
    
    return questions, answers

def evaluate_test_with_ai():
    """Evaluate test answers using AI"""
    model = setup_ai()
    if not model:
        return "AI evaluation not available"
    
    try:
        questions_text = ""
        for i, (question, user_answer) in enumerate(zip(st.session_state.test_questions, st.session_state.user_answers)):
            user_answer_text = question['options'][user_answer] if user_answer is not None else "Not answered"
            questions_text += f"Q{i+1}: {question['question']}\nUser Answer: {user_answer_text}\nCorrect Answer: {question['options'][question['correct_answer']]}\n\n"
        
        prompt = f"""
        Evaluate this student's test performance and provide detailed feedback:
        
        {questions_text}
        
        Please provide:
        1. Overall score and percentage
        2. For each incorrect answer, explain why it's wrong and why the correct answer is right
        3. Areas for improvement
        4. Encouraging feedback
        
        Be constructive and educational in your feedback.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error in AI evaluation: {str(e)}"