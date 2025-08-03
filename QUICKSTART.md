# Quick Start Guide - Enhanced Research Assistant v2.0

## 🚀 Get Started in 5 Minutes

### Step 1: Run the Setup Script

```bash
./setup.sh
```

### Step 2: Configure API Keys

Edit the `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 3: Start the Application

```bash
./start_dev.sh
```

### Step 4: Access the Application

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs

## 🎯 What You Can Do

### 1. Advanced Literature Search

- Search across multiple academic databases
- Use multilingual search capabilities
- Filter by date, author, journal, etc.

### 2. Generate Podcasts from Papers

- Convert research papers into audio podcasts
- Choose from different styles (summary, interview, debate)
- Download as MP3 files

### 3. Analyze Research Videos

- Upload or link to research videos
- Get automatic transcription and analysis
- Extract key concepts and insights

### 4. Writing Assistance

- Get help writing literature reviews
- Improve academic writing style
- Format citations automatically

### 5. Research Alerts

- Set up keyword-based alerts
- Follow specific authors
- Get notified about new papers

### 6. Collaborate with Others

- Share research projects
- Collaborate on annotations
- Get team recommendations

## 🔑 Essential API Keys

### Required

- **OpenAI API Key**: For AI features (text generation, summarization, TTS)
  - Get it at: https://platform.openai.com/

### Optional (for enhanced features)

- **Google Translate API**: For multilingual support
- **Semantic Scholar API**: For enhanced paper search
- **CrossRef API**: For citation data

## 🆘 Troubleshooting

### Common Issues

1. **Port Already in Use**

   ```bash
   # Kill processes on ports 3000 and 8000
   lsof -ti:3000 | xargs kill -9
   lsof -ti:8000 | xargs kill -9
   ```

2. **Database Connection Error**

   ```bash
   # Restart PostgreSQL container
   docker restart research_postgres
   ```

3. **Frontend Build Errors**

   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **Python Dependencies**
   ```bash
   cd backend
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Verify Installation

```bash
./test_setup.sh
```

## 📚 Next Steps

1. **Explore the Interface**: Navigate through the tabbed interface
2. **Try a Search**: Start with a simple academic query
3. **Generate Your First Podcast**: Upload a paper and create an audio summary
4. **Set Up Alerts**: Configure notifications for your research interests
5. **Invite Collaborators**: Share your research projects

## 🎓 Tips for Researchers

- Use specific, academic language in your searches
- Take advantage of the multilingual search for international papers
- Create podcasts for papers you need to review quickly
- Set up alerts for your specific research domain
- Use the writing assistant for drafting literature reviews

## 🆘 Need Help?

- Check the full README.md for detailed documentation
- Visit http://localhost:8000/docs for API documentation
- Run `./test_setup.sh` to diagnose issues

---

**Ready to enhance your research workflow!** 🎓✨
