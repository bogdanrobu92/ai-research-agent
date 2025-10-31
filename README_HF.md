# Hugging Face Spaces Deployment

Your app is ready to deploy to Hugging Face Spaces!

## Quick Deploy Steps:

1. **Create a Hugging Face account** (if you don't have one):
   - Go to https://huggingface.co/join

2. **Create a new Space**:
   - Go to https://huggingface.co/new-space
   - Name: `ai-research-agent`
   - License: MIT
   - Select SDK: **Streamlit**
   - Select visibility: **Public** (or Private)
   - Click "Create Space"

3. **Clone your Space locally**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-research-agent
   ```

4. **Copy your files**:
   ```bash
   cp app.py tools.py requirements.txt packages.txt .python-version [space-directory]/
   ```

5. **Add your OpenAI API key**:
   - In the Space settings, go to "Repository secrets"
   - Add: `OPENAI_API_KEY` = your key

6. **Push to Hugging Face**:
   ```bash
   cd [space-directory]
   git add .
   git commit -m "Initial deployment"
   git push
   ```

Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/ai-research-agent`

