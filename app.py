import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import *

# Configure page
st.set_page_config(
    page_title="SCAI - Student-Centric AI",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
def init_session_state():
    defaults = {
        'mode': 'Home',
        'chat_history_edpath': [],
        'chat_history_eddocs': [],
        'chat_history_edvision': [],
        'uploaded_doc_content': '',
        'test_questions': [],
        'test_answers': [],
        'current_question': 0,
        'user_answers': [],
        'test_started': False,
        'test_score': 0,
        'test_start_time': None,
        'test_duration': 600,
        'generated_learning_path': '',
        'test_results_calculated': False,
        'courses_generated': False,
        'current_courses': [],
        'current_topic': '',
        'uploaded_image_content': None,
        'session_start_time': datetime.now() if 'session_start_time' not in st.session_state else st.session_state.session_start_time,
        'total_questions_asked': 0,
        'total_courses_searched': 0,
        'total_documents_processed': 0,
        'total_tests_taken': 0,
        'mode_usage_count': {'Home': 0, 'EdHook': 0, 'EdPath': 0, 'EdDocs': 0, 'EdVision': 0, 'EdMocks': 0}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_form_fields():
    """Clear form fields by removing from session state"""
    keys_to_clear = ['topic_input', 'difficulty_input', 'min_rating_input']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def get_remaining_time():
    """Get remaining time for test"""
    if not st.session_state.test_start_time:
        return st.session_state.test_duration
    
    elapsed = (datetime.now() - st.session_state.test_start_time).total_seconds()
    remaining = max(0, st.session_state.test_duration - elapsed)
    return remaining

def create_chat_download(chat_history, filename_prefix):
    """Create downloadable chat history"""
    chat_text = ""
    for msg in chat_history:
        role = "User" if msg['role'] == 'user' else "AI Assistant"
        chat_text += f"{role}: {msg['content']}\n\n"
    return chat_text

def calculate_test_results():
    """Calculate test results"""
    questions = st.session_state.test_questions
    correct_answers = st.session_state.test_answers
    user_answers = st.session_state.user_answers
    
    score = 0
    for i, (correct, user) in enumerate(zip(correct_answers, user_answers)):
        if user is not None and user == correct:
            score += 1
    
    st.session_state.test_score = score
    st.session_state.test_results_calculated = True

def display_home_dashboard():
    """Enhanced Home Dashboard"""
    
    # Welcome section
    st.markdown("### Welcome to Your Learning Journey! 🚀")
    st.markdown("**SCAI** helps you discover courses, create engaging content, analyze documents, and practice with AI-powered assessments.")
    
    # Feature cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    modes = [
        ("EdHook", "🎣", "Create engaging educational hooks to capture student attention instantly"),
        ("EdPath", "🗺️", "Discover personalized learning paths with curated courses"), 
        ("EdDocs", "📚", "Upload and analyze documents with intelligent Q&A capabilities"),
        ("EdVision", "👁️", "Visual learning through image analysis and recognition"),
        ("EdMocks", "📝", "Practice with AI-generated tests and get detailed feedback")
    ]
    
    cols = [col1, col2, col3, col4, col5]
    for i, (mode, icon, desc) in enumerate(modes):
        with cols[i]:
            with st.container():
                if st.button(f"{icon} **{mode}**", key=f"dash_btn_{mode}", use_container_width=True):
                    st.session_state.mode = mode
                    st.session_state.mode_usage_count[mode] += 1
                    st.rerun()
                st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)
    
    st.divider()
    
    # Quick Stats
    st.markdown("### 📊 Your Learning Analytics")
    
    session_duration = datetime.now() - st.session_state.session_start_time
    duration_minutes = int(session_duration.total_seconds() / 60)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("⏱️ Session Time", f"{duration_minutes} min")
    with col2: st.metric("❓ Questions Asked", st.session_state.total_questions_asked)
    with col3: st.metric("📚 Courses Found", st.session_state.total_courses_searched)
    with col4: st.metric("📄 Documents Processed", st.session_state.total_documents_processed)
    with col5: st.metric("📝 Tests Taken", st.session_state.total_tests_taken)
    
    # Usage visualization - Fixed
    st.markdown("### 📈 Feature Usage This Session")
    
    # Get usage counts and filter properly
    usage_counts = st.session_state.mode_usage_count.copy()
    
    # Remove 'Home' from the counts as it's not a feature
    if 'Home' in usage_counts:
        del usage_counts['Home']
    
    # Check if there's any usage data
    total_usage = sum(usage_counts.values())
    
    if total_usage > 0:
        # Prepare data for visualization
        modes_list = []
        counts_list = []
        
        # Add all modes with their counts (including zeros for better visualization)
        feature_modes = ['EdHook', 'EdPath', 'EdDocs', 'EdVision', 'EdMocks']
        mode_labels = {
            'EdHook': '🎣 EdHook',
            'EdPath': '🗺️ EdPath', 
            'EdDocs': '📚 EdDocs',
            'EdVision': '👁️ EdVision',
            'EdMocks': '📝 EdMocks'
        }
        
        for mode in feature_modes:
            count = usage_counts.get(mode, 0)
            modes_list.append(mode_labels[mode])
            counts_list.append(count)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Feature': modes_list,
            'Usage Count': counts_list
        })
        
        # Create two types of charts based on data
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Bar chart
            fig_bar = px.bar(
                df, 
                x='Feature', 
                y='Usage Count', 
                title="Feature Usage Count",
                color='Usage Count',
                color_continuous_scale='viridis',
                text='Usage Count'
            )
            
            fig_bar.update_traces(textposition='outside')
            fig_bar.update_layout(
                showlegend=False,
                height=400,
                xaxis_title="Features",
                yaxis_title="Times Used",
                xaxis={'tickangle': 45}
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_chart2:
            # Pie chart (only for non-zero values)
            non_zero_df = df[df['Usage Count'] > 0]
            
            if len(non_zero_df) > 0:
                fig_pie = px.pie(
                    non_zero_df, 
                    values='Usage Count', 
                    names='Feature',
                    title="Usage Distribution"
                )
                
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No feature usage to display in pie chart yet!")
        
        # Usage summary
        most_used = df.loc[df['Usage Count'].idxmax()]
        if most_used['Usage Count'] > 0:
            st.success(f"🏆 Most used feature: **{most_used['Feature']}** ({most_used['Usage Count']} times)")
        
        # Display raw data in an expander for debugging
        with st.expander("📋 View Raw Usage Data"):
            st.dataframe(df)
            st.json(st.session_state.mode_usage_count)
            
    else:
        st.info("🎯 Start exploring features to see your usage analytics! Click on any feature above to begin your learning journey.")
        
        # Show placeholder chart
        placeholder_df = pd.DataFrame({
            'Feature': ['🎣 EdHook', '🗺️ EdPath', '📚 EdDocs', '👁️ EdVision', '📝 EdMocks'],
            'Usage Count': [0, 0, 0, 0, 0]
        })
        
        fig_placeholder = px.bar(
            placeholder_df, 
            x='Feature', 
            y='Usage Count', 
            title="Feature Usage (No data yet)",
            color_discrete_sequence=['lightgray']
        )
        
        fig_placeholder.update_layout(
            height=300,
            xaxis_title="Features",
            yaxis_title="Times Used",
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig_placeholder, use_container_width=True)

def display_edhook():
    """EdHook Mode - Educational Content Hooks"""
    st.header("🎣 EdHook: Educational Content Hooks")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input("📖 Enter the topic:", placeholder="e.g., Photosynthesis, Quantum Physics")
        
        col_style, col_tone = st.columns(2)
        with col_style:
            style = st.selectbox("🎨 Hook Style:", ["Story", "Question", "Surprising Fact", "Analogy", "Real-world Connection"])
        with col_tone:
            tone = st.selectbox("🎭 Tone:", ["Humorous", "Serious", "Inspiring", "Curious", "Dramatic"])
    
    with col2:
        output_size = st.selectbox("📏 Output Size:", ["Small", "Medium", "Large"])
    
    if st.button("🚀 Generate Hook", type="primary") and topic:
        with st.spinner("Generating engaging hook..."):
            hook = generate_educational_hook(topic, style, tone, output_size)
            
            st.subheader("Generated Hook:")
            st.success(hook)

def display_edpath():
    """EdPath Mode - Personalized Learning Paths"""
    st.header("🗺️ EdPath: Personalized Learning Paths")
    
    tab1, tab2 = st.tabs(["Course Discovery", "AI Learning Mentor"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)  # Changed to 4 columns
        
        with col1:
            topic = st.text_input("🔎 What would you like to learn?", placeholder="e.g., machine learning, data science", key="topic_input")
        with col2:
            difficulty = st.selectbox("📊 Course Difficulty:", ["All", "Beginner", "Intermediate", "Advanced"], key="difficulty_input")
        with col3:
            user_level = st.selectbox("👤 Your Current Level:", ["Beginner", "Intermediate", "Advanced"], key="user_level_input")
        with col4:
            min_rating = st.selectbox("⭐ Minimum Rating:", [0, 3.0, 4.0, 4.5], key="min_rating_input")
        
        search_clicked = st.button("🚀 Search Courses", type="primary")
        
        # Course Results in Grid Layout
        if search_clicked and topic:
            with st.spinner("Searching for courses..."):
                courses = search_courses_online(topic, difficulty, min_rating)
                st.session_state.current_courses = courses
                st.session_state.current_topic = topic
                st.session_state.user_difficulty = user_level  # Store user difficulty level
                st.session_state.courses_generated = True
                st.session_state.total_courses_searched += len(courses)
                
                clear_form_fields()
        
        # Display courses if they exist
        if st.session_state.courses_generated and st.session_state.current_courses:
            st.subheader("📚 Recommended Courses")
            courses = st.session_state.current_courses
            
            # Grid layout - 2 columns
            for i in range(0, len(courses), 2):
                col1, col2 = st.columns(2)
                
                # First course in row
                with col1:
                    if i < len(courses):
                        course = courses[i]
                        with st.container():
                            st.markdown(f"**📖 {course['title']}**")
                            st.markdown(f"🏫 *{course['provider']}*")
                            st.markdown(f"👨‍🏫 **Instructor:** {course.get('instructor', 'N/A')}")
                            st.markdown(f"📊 **Level:** {course['difficulty']}")
                            st.markdown(f"⭐ **Rating:** {course['rating']}")
                            st.markdown(f"{course['description'][:100]}...")
                            st.link_button("🔗 View Course", course['url'], use_container_width=True)
                            st.markdown("---")
                
                # Second course in row
                with col2:
                    if i + 1 < len(courses):
                        course = courses[i + 1]
                        with st.container():
                            st.markdown(f"**📖 {course['title']}**")
                            st.markdown(f"🏫 *{course['provider']}*")
                            st.markdown(f"👨‍🏫 **Instructor:** {course.get('instructor', 'N/A')}")
                            st.markdown(f"📊 **Level:** {course['difficulty']}")
                            st.markdown(f"⭐ **Rating:** {course['rating']}")
                            st.markdown(f"{course['description'][:100]}...")
                            st.link_button("🔗 View Course", course['url'], use_container_width=True)
                            st.markdown("---")
            
            # Generate learning path automatically after courses
            st.markdown('<h3 id="learning-path-section">🎯 Your Personalized Learning Path</h3>', unsafe_allow_html=True)
            if not st.session_state.generated_learning_path:
                with st.spinner("Creating your learning roadmap..."):
                    user_diff = st.session_state.get('user_difficulty', 'Beginner')
                    learning_path = generate_learning_path(
                        st.session_state.current_courses, 
                        st.session_state.current_topic,
                        user_difficulty=user_diff
                    )
                    st.session_state.generated_learning_path = learning_path
            
            st.markdown(st.session_state.generated_learning_path)
            
            # Download option
            st.download_button(
                label="📥 Download Learning Path",
                data=st.session_state.generated_learning_path,
                file_name=f"learning_path_{st.session_state.current_topic.replace(' ', '_')}.txt",
                mime="text/plain",
                type="secondary"
            )
    
    with tab2:
        st.subheader("🤖 AI Learning Mentor")
        
        # Background input
        background = st.text_area("📋 Describe your background:", 
                                placeholder="e.g., I'm a beginner in programming with some math background...")
        
        # Display chat history
        for msg in st.session_state.chat_history_edpath:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
        
        # Chat input
        user_input = st.text_input("💬 Ask your learning mentor:", placeholder="What should I learn to become a data scientist?")
        
        if st.button("📨 Send") and user_input:
            st.session_state.chat_history_edpath.append({"role": "user", "content": user_input})
            
            with st.spinner("AI mentor is thinking..."):
                model = setup_ai()
                if model:
                    try:
                        background_context = f"Student background: {background}\n" if background else ""
                        response = model.generate_content(f"{background_context}As an AI learning mentor, provide personalized advice for: {user_input}")
                        ai_response = response.text
                        st.session_state.chat_history_edpath.append({"role": "assistant", "content": ai_response})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

def display_eddocs():
    """EdDocs Mode - Document Analysis & Q&A"""
    st.header("📚 EdDocs: Document Analysis & Q&A")
    
    # Enhanced file uploader with more supported formats
    uploaded_file = st.file_uploader(
        "📤 Upload a document", 
        type=['txt', 'pdf', 'docx', 'doc', 'csv', 'json', 'xml', 'md', 'rtf', 'html', 
              'py', 'js', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go', 'rs', 'swift',
              'yml', 'yaml', 'ini', 'cfg', 'conf', 'toml', 'log', 'tex', 'latex'],
        help="Supports text files, documents, code files, config files, and more!"
    )
    
    if uploaded_file:
        st.success(f"📄 Document uploaded: {uploaded_file.name}")
        
        # Process document
        if 'uploaded_doc_content' not in st.session_state or st.session_state.uploaded_doc_content == '':
            with st.spinner("Processing document..."):
                st.session_state.uploaded_doc_content = process_uploaded_document(uploaded_file)
        
        # Show document preview
        with st.expander("📄 Document Preview"):
            preview_text = st.session_state.uploaded_doc_content[:1000]
            if len(st.session_state.uploaded_doc_content) > 1000:
                preview_text += "..."
            st.text(preview_text)
        
        # Chat with document
        st.subheader("💬 Chat with your document")
        
        # Display chat history
        for msg in st.session_state.chat_history_eddocs:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
        
        question = st.text_input("❓ Ask a question about the document:", placeholder="What are the main topics covered?")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            ask_clicked = st.button("🔍 Ask")
        with col2:
            if st.session_state.chat_history_eddocs:
                chat_data = create_chat_download(st.session_state.chat_history_eddocs, "document_chat")
                st.download_button("📥 Download Chat", data=chat_data, file_name="document_chat.txt", mime="text/plain")
        
        if ask_clicked and question:
            st.session_state.chat_history_eddocs.append({"role": "user", "content": question})
            
            with st.spinner("Analyzing document..."):
                answer = chat_with_document(question, st.session_state.uploaded_doc_content)
                st.session_state.chat_history_eddocs.append({"role": "assistant", "content": answer})
                st.rerun()

def display_edvision():
    """EdVision Mode - Visual Learning & Recognition"""
    st.header("👁️ EdVision: Visual Learning & Recognition")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_image = st.file_uploader(
            "📸 Upload an image", 
            type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'],
            help="Supports JPG, PNG, GIF, BMP, TIFF, and WebP formats"
        )
    
    with col2:
        operations = st.selectbox(
            "🔧 Choose analysis operation:",
            ["Describe the image", "Extract text (OCR)", "Identify objects", "Analyze educational content", "Generate questions", "Explain concepts shown"]
        )
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            
            # Display image with proper sizing
            st.image(image, caption="Uploaded Image", width=400)
            st.session_state.uploaded_image_content = image
            
            st.subheader("💬 Chat about this image")
            
            # Display chat history for this image
            for msg in st.session_state.chat_history_edvision:
                if msg['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])
            
            # Optional text input for custom questions
            question = st.text_input("❓ Ask about the image (optional):", placeholder="What do you see in this image?")
            
            # Use dropdown selection if no custom question provided
            if st.button("🔍 Analyze"):
                # Determine the question to use
                if question.strip():
                    analysis_question = question
                    user_display = question
                else:
                    # Map operations to specific instructions
                    operation_prompts = {
                        "Describe the image": "Describe what you see in this image in detail.",
                        "Extract text (OCR)": "Extract only the text visible in this image. Return just the text without any explanation, analysis or description.",
                        "Identify objects": "List only the objects you can identify in this image. Provide just the object names.",
                        "Analyze educational content": "Analyze the educational content shown in this image. Focus only on the learning material presented.",
                        "Generate questions": "Generate only questions based on what's shown in this image. Provide just the questions without explanations.",
                        "Explain concepts shown": "Explain only the concepts or subjects visible in this image. Focus on the educational explanation."
                    }
                    
                    analysis_question = operation_prompts.get(operations, operations)
                    user_display = f"Operation: {operations}"
                
                st.session_state.chat_history_edvision.append({"role": "user", "content": user_display})
                st.session_state.total_questions_asked += 1

                with st.spinner("Processing image..."):
                    result = process_image_with_ai(image, analysis_question)
                    st.session_state.chat_history_edvision.append({"role": "assistant", "content": result})
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error loading image: {str(e)}. Please ensure the file is a valid image format.")
    else:
        st.info("Upload an image to start visual analysis, and ask questions about it!")


def display_edmocktest():
    """EdMocks Mode - Practice Tests & Assessments"""
    st.header("📝 EdMocks: Practice Tests & Assessments")
    
    if not st.session_state.test_started:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            topic = st.text_input("📚 Test Topic:", placeholder="e.g., Mathematics, Science")
        with col2:
            difficulty = st.selectbox("📊 Difficulty:", ["Beginner", "Intermediate", "Advanced"])
        with col3:
            num_questions = st.number_input("❓ Number of Questions:", min_value=3, max_value=20, value=5)
        with col4:
            test_duration = st.number_input("⏱️ Duration (minutes):", min_value=1, max_value=180, value=10)
        
        if st.button("🚀 Generate Test", type="primary") and topic:
            with st.spinner("Generating test questions..."):
                questions, answers = generate_test_questions(topic, difficulty, num_questions)
                
                if questions:
                    st.session_state.test_questions = questions
                    st.session_state.test_answers = answers
                    st.session_state.user_answers = [None] * len(questions)
                    st.session_state.current_question = 0
                    st.session_state.test_started = True
                    st.session_state.test_start_time = datetime.now()
                    st.session_state.test_duration = test_duration * 60
                    st.session_state.total_tests_taken += 1
                    st.session_state.test_results_calculated = False
                    st.rerun()
                else:
                    st.error("Failed to generate questions. Please try again.")
    
    else:
        if not st.session_state.test_results_calculated:
            display_test_interface()
        else:
            display_test_results()

def display_test_interface():
    """Display test interface with timer"""
    questions = st.session_state.test_questions
    current_q = st.session_state.current_question
    
    # Timer display
    remaining_time = get_remaining_time()
    minutes = int(remaining_time // 60)
    seconds = int(remaining_time % 60)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress((current_q + 1) / len(questions))
        st.write(f"Question {current_q + 1} of {len(questions)}")
    with col2:
        if remaining_time > 0:
            st.write(f"⏱️ Time left: {minutes:02d}:{seconds:02d}")
        else:
            st.error("⏰ Time's up!")
            calculate_test_results()
            st.rerun()
    
    if current_q < len(questions) and remaining_time > 0:
        question = questions[current_q]
        
        # Question
        st.subheader(question['question'])
        
        # Options
        selected = st.radio(
            "Select your answer:",
            question['options'],
            key=f"q_{current_q}",
            index=st.session_state.user_answers[current_q] if st.session_state.user_answers[current_q] is not None else None
        )
        
        if selected:
            st.session_state.user_answers[current_q] = question['options'].index(selected)
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if current_q > 0 and st.button("⬅️ Previous"):
                st.session_state.current_question -= 1
                st.rerun()
        
        with col2:
            if current_q < len(questions) - 1 and st.button("➡️ Next"):
                st.session_state.current_question += 1
                st.rerun()
        
        with col3:
            if st.button("✅ Submit Test"):
                calculate_test_results()
                st.rerun()

def display_test_results():
    """Display test results with AI evaluation"""
    questions = st.session_state.test_questions
    score = st.session_state.test_score
    total = len(questions)
    percentage = (score / total) * 100
    
    st.header("📊 Test Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{score}/{total}")
    with col2:
        st.metric("Percentage", f"{percentage:.1f}%")
    with col3:
        grade = "Excellent" if percentage >= 80 else "Good" if percentage >= 60 else "Needs Improvement"
        st.metric("Grade", grade)
    
    # AI Evaluation
    st.subheader("🤖 AI Evaluation & Feedback")
    with st.spinner("Getting AI feedback..."):
        ai_feedback = evaluate_test_with_ai()
        st.success(ai_feedback)
        
        # Download AI feedback button
        st.download_button(
            label="📥 Download AI Feedback",
            data=ai_feedback,
            file_name="ai_evaluation_feedback.txt",
            mime="text/plain"
        )
    
    if st.button("✅ Review Done", type="primary"):
        # Reset all test-related session state
        keys_to_reset = ['test_started', 'test_questions', 'test_answers', 'user_answers', 
                        'current_question', 'test_score', 'test_start_time', 'test_results_calculated']
        for key in keys_to_reset:
            if key in st.session_state:
                if key in ['test_questions', 'test_answers', 'user_answers']:
                    st.session_state[key] = []
                elif key in ['current_question', 'test_score']:
                    st.session_state[key] = 0
                else:
                    st.session_state[key] = False if 'test_started' in key or 'calculated' in key else None
        st.rerun()

def main():
    init_session_state()
    
    col_title, col_home = st.columns([8, 1])
    with col_title:
        st.title("🎓 SCAI - Student-Centric AI")
        st.subheader("AI-Powered Educational Co-pilot for Personalized Learning")
    with col_home:
        if st.button("🏠 Home", key="home_btn"):
            st.session_state.mode = 'Home'
            st.rerun()
    
    # Mode selection 
    if st.session_state.mode != 'Home':
        st.write()
        col1, col2, col3, col4, col5 = st.columns(5)
        
        modes = [
            ("EdHook", "🎣"),
            ("EdPath", "🗺️"), 
            ("EdDocs", "📚"),
            ("EdVision", "👁️"),
            ("EdMocks", "📝")
        ]
        
        cols = [col1, col2, col3, col4, col5]
        for i, (mode, icon) in enumerate(modes):
            with cols[i]:
                if st.button(f"{icon} {mode}", key=f"btn_{mode}", use_container_width=True):
                    st.session_state.mode = mode
                    st.session_state.mode_usage_count[mode] += 1
                    st.rerun()
    
    # Display selected mode
    if st.session_state.mode == "Home":
        display_home_dashboard()
    elif st.session_state.mode == "EdHook":
        display_edhook()
    elif st.session_state.mode == "EdPath":
        display_edpath()
    elif st.session_state.mode == "EdDocs":
        display_eddocs()
    elif st.session_state.mode == "EdVision":
        display_edvision()
    elif st.session_state.mode == "EdMocks":
        display_edmocktest()

if __name__ == "__main__":
    main()