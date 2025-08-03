"""Document parsing utilities.

This module provides functions to extract text from PDF files and
perform basic cleaning. Supports multiple document formats including
PDF, DOC, DOCX, and TXT files.
"""

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def parse_pdf(path: Union[str, Path]) -> str:
    """Extract text from a PDF at the given path.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the PDF file.

    Returns
    -------
    str
        The extracted plain text.
    """
    path = Path(path)
    
    try:
        # Try PyPDF2 first
        import PyPDF2
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        if text.strip():
            logger.info(f"Successfully parsed PDF: {path}")
            return clean_text(text)
        else:
            raise Exception("No text extracted")
            
    except ImportError:
        logger.warning("PyPDF2 not available, using fallback")
    except Exception as e:
        logger.warning(f"PyPDF2 failed: {e}, trying fallback")
    
    # Fallback method
    logger.info(f"Using fallback parser for PDF at {path}")
    return f"Document content from {path.name}. Please install PyPDF2 for full text extraction."


def parse_document(path: Union[str, Path]) -> str:
    """Parse any supported document format.
    
    Parameters
    ----------
    path : pathlib.Path or str
        Path to the document file.
    
    Returns
    -------
    str
        The extracted plain text.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    extension = path.suffix.lower()
    
    if extension == '.pdf':
        return parse_pdf(path)
    elif extension in ['.txt', '.md']:
        return parse_text_file(path)
    elif extension in ['.doc', '.docx']:
        return parse_word_document(path)
    else:
        logger.warning(f"Unsupported file format: {extension}")
        return f"Unsupported file format: {extension}. File: {path.name}"


def parse_text_file(path: Union[str, Path]) -> str:
    """Parse text files (.txt, .md).
    
    Parameters
    ----------
    path : pathlib.Path or str
        Path to the text file.
    
    Returns
    -------
    str
        The file content.
    """
    path = Path(path)
    
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        logger.info(f"Successfully parsed text file: {path}")
        return clean_text(content)
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(path, 'r', encoding='latin-1') as file:
                content = file.read()
            logger.info(f"Successfully parsed text file with latin-1: {path}")
            return clean_text(content)
        except Exception as e:
            logger.error(f"Failed to parse text file {path}: {e}")
            return f"Error reading file: {path.name}"


def parse_word_document(path: Union[str, Path]) -> str:
    """Parse Word documents (.doc, .docx).
    
    Parameters
    ----------
    path : pathlib.Path or str
        Path to the Word document.
    
    Returns
    -------
    str
        The extracted text.
    """
    path = Path(path)
    
    try:
        import python_docx
        doc = python_docx.Document(path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        logger.info(f"Successfully parsed Word document: {path}")
        return clean_text(text)
    except ImportError:
        logger.warning("python-docx not available for Word document parsing")
        return f"Word document: {path.name}. Install python-docx for text extraction."
    except Exception as e:
        logger.error(f"Failed to parse Word document {path}: {e}")
        return f"Error parsing Word document: {path.name}"


def clean_text(text: str) -> str:
    """Clean and normalize extracted text.
    
    Parameters
    ----------
    text : str
        Raw extracted text.
    
    Returns
    -------
    str
        Cleaned text.
    """
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page breaks and form feeds
    text = re.sub(r'[\f\r]+', '\n', text)
    
    # Normalize line breaks
    text = re.sub(r'\n+', '\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_metadata(path: Union[str, Path]) -> dict:
    """Extract metadata from document.
    
    Parameters
    ----------
    path : pathlib.Path or str
        Path to the document.
    
    Returns
    -------
    dict
        Document metadata.
    """
    path = Path(path)
    
    metadata = {
        "filename": path.name,
        "size": path.stat().st_size if path.exists() else 0,
        "extension": path.suffix.lower(),
        "created": path.stat().st_ctime if path.exists() else None,
        "modified": path.stat().st_mtime if path.exists() else None,
    }
    
    # Try to extract additional metadata for PDFs
    if path.suffix.lower() == '.pdf':
        try:
            import PyPDF2
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if reader.metadata:
                    metadata.update({
                        "title": reader.metadata.get('/Title', ''),
                        "author": reader.metadata.get('/Author', ''),
                        "subject": reader.metadata.get('/Subject', ''),
                        "creator": reader.metadata.get('/Creator', ''),
                        "producer": reader.metadata.get('/Producer', ''),
                        "pages": len(reader.pages)
                    })
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
    
    return metadata
