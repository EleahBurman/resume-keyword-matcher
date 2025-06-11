# Import Flask framework and utilities
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename

# Import our custom utility modules (we'll create these next)
from utils.file_handler import FileHandler
from utils.text_processor import TextProcessor
from utils.matcher import KeywordMatcher

# Create Flask application instance
app = Flask(__name__)

# Configuration settings
app.config['SECRET_KEY'] = 'your-secret-key-change-this'  # Change this in production!
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Where uploaded files are temporarily stored
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size limit

# Ensure the upload directory exists (create if it doesn't)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define which file types we'll accept for upload
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}

def allowed_file(filename):
    """
    Check if uploaded file has an allowed extension
    Args:
        filename (str): Name of the uploaded file
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """
    Home page route - displays the main upload form
    Users can upload resume/job description files or paste text directly
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Handle file uploads and process keyword matching
    This is the main processing route that:
    1. Accepts resume and job description (as files or text)
    2. Extracts text from uploaded files
    3. Processes the text to find keywords
    4. Compares keywords and calculates match score
    5. Returns results page with highlighted matches
    """
    try:
        # Initialize our utility classes
        file_handler = FileHandler()        # Handles file reading/text extraction
        text_processor = TextProcessor()    # Processes text and extracts keywords
        matcher = KeywordMatcher()          # Compares keywords and calculates scores
        
        # === PROCESS RESUME INPUT ===
        resume_text = ""
        
        # Check if user uploaded a resume file
        if 'resume_file' in request.files and request.files['resume_file'].filename:
            resume_file = request.files['resume_file']
            
            # Verify file type is allowed
            if allowed_file(resume_file.filename):
                # Secure the filename to prevent directory traversal attacks
                filename = secure_filename(resume_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save file temporarily, extract text, then delete
                resume_file.save(filepath)
                resume_text = file_handler.extract_text(filepath)
                os.remove(filepath)  # Clean up temporary file
            else:
                flash('Invalid resume file type. Please upload PDF, DOCX, or TXT files.')
                return redirect(url_for('index'))
                
        # If no file uploaded, check if user pasted text directly
        elif request.form.get('resume_text'):
            resume_text = request.form.get('resume_text')
        
        # === PROCESS JOB DESCRIPTION INPUT ===
        job_desc_text = ""
        
        # Check if user uploaded a job description file
        if 'job_desc_file' in request.files and request.files['job_desc_file'].filename:
            job_desc_file = request.files['job_desc_file']
            
            # Verify file type is allowed
            if allowed_file(job_desc_file.filename):
                filename = secure_filename(job_desc_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Save file temporarily, extract text, then delete
                job_desc_file.save(filepath)
                job_desc_text = file_handler.extract_text(filepath)
                os.remove(filepath)  # Clean up temporary file
            else:
                flash('Invalid job description file type. Please upload PDF, DOCX, or TXT files.')
                return redirect(url_for('index'))
                
        # If no file uploaded, check if user pasted text directly
        elif request.form.get('job_desc_text'):
            job_desc_text = request.form.get('job_desc_text')
        
        # === VALIDATE INPUT ===
        # Make sure we have both resume and job description text
        if not resume_text.strip() or not job_desc_text.strip():
            flash('Please provide both resume and job description.')
            return redirect(url_for('index'))
        
        # === PROCESS THE TEXT ===
        # Extract important keywords from both documents
        resume_keywords = text_processor.extract_keywords(resume_text)
        job_desc_keywords = text_processor.extract_keywords(job_desc_text)
        
        # === PERFORM MATCHING ===
        # Compare keywords and calculate compatibility score
        results = matcher.match_keywords(resume_keywords, job_desc_keywords)
        
        # Add original texts to results so we can highlight matches in the UI
        results['resume_text'] = resume_text
        results['job_desc_text'] = job_desc_text
        
        # Display results page with match score and highlighted text
        return render_template('results.html', results=results)
        
    except Exception as e:
        # If anything goes wrong, show error message and return to home page
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    API endpoint for keyword analysis (for programmatic access)
    Accepts JSON data with resume_text and job_desc_text fields
    Returns JSON response with matching results
    This allows other applications to use our matching functionality
    """
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Validate that required fields are present
        if not data or 'resume_text' not in data or 'job_desc_text' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Initialize processors
        text_processor = TextProcessor()
        matcher = KeywordMatcher()
        
        # Extract keywords from both texts
        resume_keywords = text_processor.extract_keywords(data['resume_text'])
        job_desc_keywords = text_processor.extract_keywords(data['job_desc_text'])
        
        # Perform matching and return results as JSON
        results = matcher.match_keywords(resume_keywords, job_desc_keywords)
        
        return jsonify(results)
        
    except Exception as e:
        # Return error as JSON response
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """
    Handle file upload size errors
    Flask automatically returns 413 error when file exceeds MAX_CONTENT_LENGTH
    This provides a user-friendly error message instead of a generic error page
    """
    flash('File is too large. Please upload files smaller than 16MB.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    """
    Run the Flask development server
    - debug=True: Enables auto-reload when code changes and detailed error pages
    - host='0.0.0.0': Makes server accessible from other devices on your network
    - port=5000: The port the server will run on (access via http://localhost:5000)
    
    Note: Only use debug=True in development, never in production!
    """
    app.run(debug=True, host='0.0.0.0', port=5001)