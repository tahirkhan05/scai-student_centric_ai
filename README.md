# 🎓 SCAI - Student-Centric AI

**AI-Powered Educational Co-pilot for Personalized Learning**

SCAI is a comprehensive educational platform that leverages Google's Gemini AI to provide students with personalized learning experiences across multiple domains.


![Video Description](scai-demo.mkv)

## ✨ Features

### 🎣 EdHook - Educational Content Hooks
- Generate engaging educational content to capture student attention
- Multiple styles: Story, Question, Surprising Fact, Analogy, Real-world Connection
- Customizable tone and output size
- Perfect for teachers and content creators

### 🗺️ EdPath - Personalized Learning Paths
- **Course Discovery**: Search and find online courses from major platforms
- **AI Learning Mentor**: Get personalized advice and learning roadmaps
- Curated courses from Coursera, edX, Udemy, Khan Academy, and more
- Custom learning paths based on your background and goals

### 📚 EdDocs - Document Analysis & Q&A
- Upload and analyze various document formats (PDF, DOCX, TXT, and more)
- Intelligent Q&A with your documents using RAG approach
- Support for code files, configuration files, and academic papers
- Chat interface for interactive document exploration

### 👁️ EdVision - Visual Learning & Recognition
- Image analysis and recognition using Gemini Vision
- Educational content extraction from images
- OCR (Optical Character Recognition) capabilities
- Generate questions from visual content

### 📝 EdMocks - Practice Tests & Assessments
- AI-generated practice tests on any topic
- Customizable difficulty levels and question counts
- Timed assessments with progress tracking
- Detailed AI-powered feedback and evaluation

### 📊 Analytics Dashboard
- Session time tracking
- Questions asked counter
- Documents processed metrics
- Feature usage analytics

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google AI API key (Gemini)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/tahirkhan05/scai-student_centric_ai
```

2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

5. **Run the application**
```bash
streamlit run app.py
```

## 📦 Dependencies

### Core Dependencies
- **streamlit** - Web application framework
- **google-generativeai** - Google Gemini AI integration
- **langchain-google-genai** - LangChain integration with Google AI
- **PIL (Pillow)** - Image processing
- **plotly** - Interactive visualizations
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### Document Processing
- **PyPDF2** - PDF document processing
- **python-docx** - DOCX document processing

### Additional Libraries
- **requests** - HTTP requests
- **python-dotenv** - Environment variable management
- **langchain** - LLM framework components

## 🔧 Configuration

### Google AI API Setup
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add the key to your `.env` file as `GOOGLE_API_KEY`

## 🔒 Privacy & Security

- No data is stored permanently
- Session-based storage only
- Documents processed locally
- API calls made securely to Google AI

## 🚨 Limitations

- Requires active internet connection
- Google AI API rate limits apply
- Large documents may take longer to process
- Image processing depends on Google Vision API availability

## 🤝 Contributing

Fork the repository, Create a feature branch, Make your changes, Test thoroughly, Submit a pull request

**Made with ❤️ for the educational community**
