# Enhanced Text Analytics AI Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-4.25+-orange.svg)](https://gradio.app/)



<img width="1690" height="1166" alt="Screenshot 2025-09-17 at 12 48 08" src="https://github.com/user-attachments/assets/36fd8b47-be28-409f-a9c8-26fc03fe7cf0" />


A comprehensive, multi-modal text analytics platform that combines smart column detection, advanced NLP processing, and multiple AI model integrations to provide actionable insights from customer feedback and text data.

## Features

### Core Analytics
- **Smart Column Detection**: Automatically identifies text, ID, product, and date columns
- **Sentiment Analysis**: TextBlob-powered sentiment classification with numerical scoring
- **Topic Extraction**: Multi-level topic identification using noun phrases and frequency analysis
- **Actionable Insights**: Dictionary-based extraction of improvement suggestions
- **Advanced Search**: TF-IDF vectorized semantic search with synonym expansion

### AI Model Integration
- **Multi-Provider Support**: OpenAI, Anthropic Claude, Deepseek, Groq, Google Gemini
- **Dynamic Model Switching**: Change AI models on-the-fly
- **Unified Interface**: Consistent API across different providers
- **AI-Powered Insights**: Generate high-level analysis using selected AI models

### Data Processing
- **Memory Efficient**: Smart data extraction and garbage collection
- **Multiple Formats**: Support for CSV, Excel (.xlsx, .xls), and JSON files
- **Batch Processing**: Handle large datasets efficiently
- **Export Options**: Export results in Excel or CSV formats

### Visualization & Interface
- **Interactive Charts**: Plotly-powered visualizations
- **Web Interface**: User-friendly Gradio-based UI
- **Real-time Processing**: Live feedback during data processing
- **Multiple Views**: Sentiment distribution, topic analysis, trends over time

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/text-analytics-ai-agent.git
cd text-analytics-ai-agent
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('brown')
```

### Step 5: Download TextBlob Corpora

```bash
python -m textblob.download_corpora
```

## Configuration

### Environment Variables

Create a `.env` file in the project root directory:

```env
# AI Model API Keys (add only the ones you plan to use)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### Obtaining API Keys

| Provider | How to Get API Key | Documentation |
|----------|-------------------|---------------|
| **OpenAI** | [OpenAI Platform](https://platform.openai.com/api-keys) | [OpenAI Docs](https://platform.openai.com/docs) |
| **Anthropic** | [Anthropic Console](https://console.anthropic.com/) | [Anthropic Docs](https://docs.anthropic.com/) |
| **Deepseek** | [Deepseek Platform](https://platform.deepseek.com/) | [Deepseek Docs](https://platform.deepseek.com/docs) |
| **Groq** | [Groq Console](https://console.groq.com/) | [Groq Docs](https://console.groq.com/docs) |
| **Google** | [Google AI Studio](https://aistudio.google.com/) | [Gemini Docs](https://ai.google.dev/docs) |

**Note**: The system works with any combination of API keys. You don't need all providers configured.

## Quick Start

### Basic Usage

1. **Start the Application**:
   ```bash
   python Multimodal_Text_Analytics.py
   ```

2. **Open the Web Interface**: 
   - The application will provide a local URL (typically `http://127.0.0.1:7860`)
   - A public sharing link will also be generated automatically

3. **Upload Your Data**:
   - Select an AI model from the dropdown
   - Upload a CSV, Excel, or JSON file containing text data
   - Click "Process File"

4. **Explore Results**:
   - View processing status and AI insights
   - Search through your data
   - Generate visualizations
   - Export processed results

### Sample Data Format

Your data should contain text columns (comments, feedback, reviews, etc.). The system automatically detects:

```csv
id,customer_feedback,product_name,date,rating
1,"Great product but delivery was slow","Widget A","2024-01-15",4
2,"Poor quality, broke after one day","Widget B","2024-01-16",1
3,"Excellent customer service, very helpful","Service","2024-01-17",5
```

## Usage Guide

### 1. Upload & Process Tab

**File Upload**:
- Supported formats: `.csv`, `.xlsx`, `.xls`, `.json`
- Automatic column detection for text, ID, product, and date fields
- Memory-efficient processing with progress feedback

**AI Model Selection**:
- Choose from available AI providers
- Switch models dynamically
- Generate AI-powered insights from processed data

**Processing Results**:
- Smart column detection summary
- Data preview (first 10 rows)
- Downloadable processed file with analysis columns

### 2. Search Tab

**Semantic Search**:
- Enter keywords to find relevant text entries
- Synonym expansion for better matching
- Similarity scoring with exact match boosting
- Export search results

**Search Features**:
- TF-IDF vectorized search
- Cosine similarity ranking
- Multi-term query support
- Results include sentiment and topics

### 3. Visualizations Tab

**Available Visualizations**:
- **Sentiment Distribution**: Pie chart of positive/negative/neutral sentiment
- **Topic Distribution**: Bar chart of most common topics
- **Sentiment by Topic**: Heatmap showing sentiment patterns across topics
- **Sentiment Timeline**: Trend analysis over time (if date data available)
- **Top Insights**: Most frequent actionable insights

**Interactive Features**:
- Plotly-powered interactive charts
- Zoom, pan, and hover functionality
- Downloadable chart images

### 4. Export Tab

**Export Options**:
- **Excel Format**: Full analysis with formatting
- **CSV Format**: Lightweight, compatible format
- **Timestamp**: Automatic file naming with timestamps

**Export Contents**:
- Original data plus analysis columns
- Sentiment scores and classifications
- Extracted topics (3 levels)
- Actionable insights
- Search scores (if applicable)

## Architecture

### Core Components

```
├── SmartColumnDetector     # Automatic column type detection
├── EnhancedTextProcessor   # NLP processing and insights extraction
├── TextSearchEngine        # Advanced search with semantic capabilities
├── AIModelManager         # Multi-provider AI model integration
└── EnhancedTextAnalyzer   # Main orchestration class
```

### Data Flow

1. **File Upload** → Smart column detection → Data extraction
2. **Text Processing** → Sentiment analysis → Topic extraction → Insights generation
3. **Search Index** → TF-IDF vectorization → Similarity calculations
4. **AI Analysis** → Sample selection → Prompt generation → Insight generation
5. **Visualization** → Data aggregation → Chart generation → Interactive display

### Processing Pipeline

```
Raw Data → Column Detection → Text Cleaning → Sentiment Analysis → 
Topic Extraction → Insights Generation → Search Index → 
Visualizations → Export Options
```

## Contributing

We welcome contributions! Here's how you can help:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run existing tests: `python -m pytest tests/`
6. Commit changes: `git commit -am 'Add feature'`
7. Push to branch: `git push origin feature-name`
8. Create a Pull Request

### Contribution Areas

- **New AI Providers**: Add support for additional AI APIs
- **Enhanced NLP**: Improve topic extraction and sentiment analysis
- **Visualizations**: Create new chart types and insights
- **Performance**: Optimize processing for larger datasets
- **Documentation**: Improve guides and examples
- **Testing**: Add comprehensive test coverage

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Add docstrings for all functions and classes
- Include inline comments for complex logic

## Troubleshooting

### Common Issues

**1. NLTK Data Missing**
```
Error: Resource punkt not found
```
**Solution**: Run the NLTK download commands in the installation section.

**2. TextBlob Corpora Missing**
```
Error: Resource 'corpora/brown' not found
```
**Solution**: Run `python -m textblob.download_corpora`

**3. API Key Issues**
```
Error: No API key provided
```
**Solution**: Check your `.env` file configuration and ensure API keys are valid.

**4. Memory Issues with Large Files**
```
MemoryError: Unable to allocate array
```
**Solution**: Process files in smaller chunks or increase system memory.

**5. Gradio Port Conflicts**
```
Error: Port 7860 is already in use
```
**Solution**: The application will automatically find an available port.

### Performance Optimization

**For Large Datasets**:
- Process files with < 50,000 rows for optimal performance
- Use CSV format for faster loading
- Close unnecessary applications to free memory

**For Slow AI Responses**:
- Check internet connection
- Verify API key limits haven't been exceeded
- Try switching to a different AI provider

### Getting Help

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check this README and inline code comments
- **Community**: Join discussions in the Issues section

## Performance Benchmarks

| Dataset Size | Processing Time | Memory Usage | Recommended |
|--------------|----------------|--------------|-------------|
| 1K rows      | 10-30 seconds  | 200MB       | Optimal     |
| 10K rows     | 1-3 minutes    | 500MB       | Good        |
| 50K rows     | 5-15 minutes   | 1.5GB       | Caution     |
| 100K+ rows   | 15+ minutes    | 3GB+        | Consider chunking |

## Roadmap

### Version 2.0 (Planned)
- [ ] Real-time data streaming support
- [ ] Custom AI model integration
- [ ] Advanced topic modeling (LDA, BERTopic)
- [ ] Multi-language support
- [ ] API endpoint for programmatic access

### Version 2.1 (Future)
- [ ] Automated report generation
- [ ] Integration with business intelligence tools
- [ ] Custom visualization builder
- [ ] Advanced export options (PDF reports)
- [ ] User authentication and data persistence

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

- **Gradio Team**: For the web interface framework
- **Hugging Face**: For NLP tools and model hosting
- **Plotly**: For interactive visualization capabilities
- **NLTK Team**: For comprehensive natural language processing tools
- **TextBlob**: For sentiment analysis capabilities
- **scikit-learn**: For machine learning algorithms and utilities

## Support

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/text-analytics-ai-agent/issues)
- **Documentation**: [Wiki](https://github.com/yourusername/text-analytics-ai-agent/wiki)

---

**Made for the open source community**
