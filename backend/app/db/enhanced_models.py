"""Enhanced database models for the Research Assistant."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Boolean,
    Float,
    ForeignKey,
    JSON,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Paper(Base):
    """Enhanced database model for research papers."""

    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    abstract = Column(Text)
    authors = Column(JSON)  # List of author names
    doi = Column(String, unique=True, index=True)
    arxiv_id = Column(String, index=True)
    pubmed_id = Column(String, index=True)
    publication_date = Column(DateTime)
    journal = Column(String)
    venue = Column(String)
    citation_count = Column(Integer, default=0)
    h_index_contribution = Column(Float, default=0.0)
    categories = Column(JSON)  # List of categories
    keywords = Column(JSON)  # List of keywords
    url = Column(String)
    pdf_url = Column(String)
    source = Column(String)  # arxiv, pubmed, etc.
    language = Column(String, default="en")

    # Advanced analysis fields
    novelty_score = Column(Float)
    impact_prediction = Column(Float)
    methodology_type = Column(String)
    research_field = Column(String)
    country_affiliation = Column(String)

    # Processing status
    processed = Column(Boolean, default=False)
    summary_generated = Column(Boolean, default=False)
    embedding_generated = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Researcher(Base):
    """Enhanced database model for researchers."""

    __tablename__ = "researchers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    affiliation = Column(String)
    email = Column(String)
    google_scholar_id = Column(String, unique=True)
    orcid = Column(String, unique=True)
    researchgate_id = Column(String)
    linkedin_url = Column(String)
    homepage = Column(String)

    # Metrics
    h_index = Column(Integer, default=0)
    h_index_5y = Column(Integer, default=0)
    i10_index = Column(Integer, default=0)
    i10_index_5y = Column(Integer, default=0)
    citation_count = Column(Integer, default=0)
    citation_count_5y = Column(Integer, default=0)

    # Research information
    research_interests = Column(JSON)  # List of research areas
    recent_papers = Column(JSON)  # List of paper IDs
    collaboration_network = Column(JSON)  # Network data
    active_grants = Column(JSON)  # List of grants

    # Location and demographics
    country = Column(String)
    institution_type = Column(String)  # university, industry, government
    career_stage = Column(String)  # student, postdoc, faculty, etc.

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ResearchProject(Base):
    """Database model for research projects."""

    __tablename__ = "research_projects"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    research_questions = Column(JSON)  # List of questions
    methodology = Column(Text)
    status = Column(String, default="planning")  # planning, active, completed, paused
    related_papers = Column(JSON)  # List of paper IDs
    collaborators = Column(JSON)  # List of researcher IDs
    funding_sources = Column(JSON)  # List of grants/funding
    deadlines = Column(JSON)  # Important dates

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Citation(Base):
    """Database model for paper citations."""

    __tablename__ = "citations"

    id = Column(Integer, primary_key=True, index=True)
    citing_paper_id = Column(Integer, ForeignKey("papers.id"))
    cited_paper_id = Column(Integer, ForeignKey("papers.id"))
    context = Column(Text)  # Citation context
    citation_type = Column(String)  # background, method, comparison, etc.

    citing_paper = relationship("Paper", foreign_keys=[citing_paper_id])
    cited_paper = relationship("Paper", foreign_keys=[cited_paper_id])

    created_at = Column(DateTime, default=datetime.utcnow)


class UserLibrary(Base):
    """Database model for user's personal library."""

    __tablename__ = "user_library"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"))
    status = Column(String, default="to_read")  # to_read, reading, read, archived
    notes = Column(Text)
    rating = Column(Integer)  # 1-5 stars
    tags = Column(JSON)  # User-defined tags
    reading_progress = Column(Float, default=0.0)  # 0-100%

    paper = relationship("Paper")

    added_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PodcastEpisode(Base):
    """Database model for generated podcast episodes."""

    __tablename__ = "podcast_episodes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    script = Column(Text)  # Full podcast script
    audio_url = Column(String)  # URL to audio file
    duration = Column(Integer)  # Duration in seconds
    paper_ids = Column(JSON)  # Papers discussed in episode
    speaker_voices = Column(JSON)  # Voice configuration
    language = Column(String, default="en")

    # Metadata
    episode_type = Column(String)  # summary, interview, discussion
    difficulty_level = Column(String)  # beginner, intermediate, advanced
    topics = Column(JSON)  # List of topics covered

    generated_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="generated")  # generated, published, archived


class VideoAnalysis(Base):
    """Database model for video analysis results."""

    __tablename__ = "video_analyses"

    id = Column(Integer, primary_key=True, index=True)
    video_url = Column(String, nullable=False)
    title = Column(String)
    description = Column(Text)
    transcript = Column(Text)
    summary = Column(Text)
    key_points = Column(JSON)  # List of key points
    topics = Column(JSON)  # List of topics
    mentioned_papers = Column(JSON)  # Referenced papers

    # Analysis metadata
    duration = Column(Integer)  # Duration in seconds
    language = Column(String)
    confidence_score = Column(Float)  # Transcription confidence

    created_at = Column(DateTime, default=datetime.utcnow)


class ResearchAlert(Base):
    """Database model for research alerts."""

    __tablename__ = "research_alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    alert_type = Column(String, nullable=False)  # keyword, author, journal, topic
    query = Column(String, nullable=False)
    frequency = Column(String, default="weekly")  # daily, weekly, monthly
    active = Column(Boolean, default=True)

    # Alert configuration
    filters = Column(JSON)  # Additional filters
    notification_method = Column(String, default="email")  # email, webhook, in-app
    last_triggered = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CollaborationRequest(Base):
    """Database model for collaboration requests."""

    __tablename__ = "collaboration_requests"

    id = Column(Integer, primary_key=True, index=True)
    requester_id = Column(String, nullable=False)
    target_researcher_id = Column(Integer, ForeignKey("researchers.id"))
    project_id = Column(Integer, ForeignKey("research_projects.id"), nullable=True)

    message = Column(Text)
    status = Column(String, default="pending")  # pending, accepted, declined, expired
    collaboration_type = Column(String)  # research, review, mentorship

    target_researcher = relationship("Researcher")
    project = relationship("ResearchProject")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WritingAssistance(Base):
    """Database model for writing assistance sessions."""

    __tablename__ = "writing_assistance"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    document_title = Column(String)
    original_text = Column(Text)
    improved_text = Column(Text)
    suggestions = Column(JSON)  # List of suggestions
    metrics = Column(JSON)  # Readability scores, etc.
    assistance_type = Column(String)  # grammar, style, structure, citations

    created_at = Column(DateTime, default=datetime.utcnow)


class IntegrityCheck(Base):
    """Database model for academic integrity checks."""

    __tablename__ = "integrity_checks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    document_title = Column(String)
    content = Column(Text)

    # Check results
    plagiarism_score = Column(Float)  # 0-100%
    ai_detection_score = Column(Float)  # 0-100%
    similarity_matches = Column(JSON)  # List of similar content
    ai_detection_details = Column(JSON)  # Details about AI detection

    # Metadata
    check_type = Column(String)  # plagiarism, ai_detection, both
    status = Column(String, default="completed")  # pending, completed, failed

    created_at = Column(DateTime, default=datetime.utcnow)
