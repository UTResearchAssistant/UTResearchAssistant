"""Enhanced Django models for the research assistant.

This module defines all the database models for the enhanced research assistant
including papers, researchers, podcasts, videos, alerts, collaboration, and more.
"""

from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import JSONField
from django.utils import timezone
import uuid


class BaseModel(models.Model):
    """Base model with common fields for all models."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Researcher(BaseModel):
    """Model for researcher profiles."""

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    orcid_id = models.CharField(max_length=19, blank=True, null=True)
    affiliation = models.CharField(max_length=200, blank=True)
    research_interests = models.TextField(blank=True)
    h_index = models.IntegerField(null=True, blank=True)
    citation_count = models.IntegerField(default=0)
    bio = models.TextField(blank=True)
    website = models.URLField(blank=True)
    profile_image = models.ImageField(upload_to="profiles/", blank=True, null=True)

    def __str__(self):
        return f"{self.user.get_full_name()} - {self.affiliation}"


class Paper(BaseModel):
    """Model for research papers."""

    title = models.CharField(max_length=500)
    abstract = models.TextField()
    authors = models.TextField()  # JSON string of author list
    doi = models.CharField(max_length=100, unique=True, null=True, blank=True)
    arxiv_id = models.CharField(max_length=50, null=True, blank=True)
    pubmed_id = models.CharField(max_length=50, null=True, blank=True)
    publication_date = models.DateField(null=True, blank=True)
    journal = models.CharField(max_length=200, blank=True)
    volume = models.CharField(max_length=20, blank=True)
    issue = models.CharField(max_length=20, blank=True)
    pages = models.CharField(max_length=50, blank=True)
    pdf_url = models.URLField(blank=True)
    external_url = models.URLField(blank=True)
    keywords = models.TextField(blank=True)  # JSON string
    subject_categories = models.TextField(blank=True)  # JSON string
    citation_count = models.IntegerField(default=0)
    language = models.CharField(max_length=10, default="en")
    is_open_access = models.BooleanField(default=False)
    content = models.TextField(blank=True)  # Full text content
    embeddings = models.TextField(blank=True)  # Vector embeddings
    summary = models.TextField(blank=True)

    # Metadata
    source = models.CharField(max_length=50, default="manual")  # arxiv, pubmed, etc.
    quality_score = models.FloatField(null=True, blank=True)
    relevance_score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.title[:100]

    class Meta:
        ordering = ["-publication_date", "-created_at"]


class PodcastEpisode(BaseModel):
    """Model for generated podcast episodes."""

    title = models.CharField(max_length=200)
    description = models.TextField()
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE, related_name="podcasts")
    creator = models.ForeignKey(User, on_delete=models.CASCADE)

    # Podcast details
    style = models.CharField(
        max_length=50,
        choices=[
            ("summary", "Summary Style"),
            ("interview", "Interview Style"),
            ("debate", "Debate Style"),
            ("educational", "Educational Style"),
        ],
        default="summary",
    )
    duration_seconds = models.IntegerField()
    audio_file = models.FileField(upload_to="podcasts/", null=True, blank=True)
    transcript = models.TextField(blank=True)
    script = models.TextField(blank=True)

    # Generation metadata
    voice_model = models.CharField(max_length=50, default="alloy")
    language = models.CharField(max_length=10, default="en")
    generation_time = models.FloatField(null=True, blank=True)
    file_size = models.BigIntegerField(null=True, blank=True)

    # Engagement metrics
    play_count = models.IntegerField(default=0)
    like_count = models.IntegerField(default=0)
    share_count = models.IntegerField(default=0)

    def __str__(self):
        return f"Podcast: {self.title}"

    class Meta:
        ordering = ["-created_at"]


class VideoAnalysis(BaseModel):
    """Model for video analysis results."""

    title = models.CharField(max_length=200)
    video_url = models.URLField()
    creator = models.ForeignKey(User, on_delete=models.CASCADE)

    # Video metadata
    duration_seconds = models.IntegerField(null=True, blank=True)
    video_type = models.CharField(
        max_length=50,
        choices=[
            ("lecture", "Academic Lecture"),
            ("conference", "Conference Presentation"),
            ("seminar", "Research Seminar"),
            ("interview", "Research Interview"),
            ("tutorial", "Educational Tutorial"),
        ],
        default="lecture",
    )
    language = models.CharField(max_length=10, default="en")

    # Analysis results
    transcript = models.TextField(blank=True)
    summary = models.TextField(blank=True)
    key_concepts = models.TextField(blank=True)  # JSON string
    timeline = models.TextField(blank=True)  # JSON string
    topics = models.TextField(blank=True)  # JSON string
    sentiment_analysis = models.TextField(blank=True)  # JSON string

    # Processing metadata
    processing_status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
        default="pending",
    )
    error_message = models.TextField(blank=True)
    processing_time = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Video Analysis: {self.title}"

    class Meta:
        ordering = ["-created_at"]


class ResearchAlert(BaseModel):
    """Model for research alerts and notifications."""

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)

    # Alert configuration
    alert_type = models.CharField(
        max_length=50,
        choices=[
            ("keyword", "Keyword Alert"),
            ("author", "Author Alert"),
            ("journal", "Journal Alert"),
            ("citation", "Citation Alert"),
            ("conference", "Conference Alert"),
        ],
    )
    keywords = models.TextField(blank=True)  # JSON string
    authors = models.TextField(blank=True)  # JSON string
    journals = models.TextField(blank=True)  # JSON string

    # Alert settings
    frequency = models.CharField(
        max_length=20,
        choices=[
            ("daily", "Daily"),
            ("weekly", "Weekly"),
            ("monthly", "Monthly"),
            ("immediate", "Immediate"),
        ],
        default="weekly",
    )
    is_active = models.BooleanField(default=True)
    last_triggered = models.DateTimeField(null=True, blank=True)
    trigger_count = models.IntegerField(default=0)

    # Notification preferences
    email_notifications = models.BooleanField(default=True)
    web_notifications = models.BooleanField(default=True)
    max_results = models.IntegerField(default=10)

    def __str__(self):
        return f"Alert: {self.title} ({self.user.username})"

    class Meta:
        ordering = ["-created_at"]


class CollaborationRequest(BaseModel):
    """Model for research collaboration requests."""

    requester = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="sent_requests"
    )
    recipient = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="received_requests"
    )

    # Collaboration details
    title = models.CharField(max_length=200)
    description = models.TextField()
    collaboration_type = models.CharField(
        max_length=50,
        choices=[
            ("research", "Research Project"),
            ("review", "Paper Review"),
            ("coauthoring", "Co-authoring"),
            ("mentoring", "Mentoring"),
            ("data_sharing", "Data Sharing"),
        ],
    )

    # Status and timeline
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("accepted", "Accepted"),
            ("declined", "Declined"),
            ("completed", "Completed"),
            ("cancelled", "Cancelled"),
        ],
        default="pending",
    )
    deadline = models.DateField(null=True, blank=True)
    response_message = models.TextField(blank=True)
    responded_at = models.DateTimeField(null=True, blank=True)

    # Associated resources
    papers = models.ManyToManyField(Paper, blank=True)
    shared_data = models.TextField(blank=True)  # JSON string

    def __str__(self):
        return f"Collaboration: {self.title} ({self.requester.username} → {self.recipient.username})"

    class Meta:
        ordering = ["-created_at"]


class WritingAssistance(BaseModel):
    """Model for writing assistance sessions."""

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)

    # Writing task details
    task_type = models.CharField(
        max_length=50,
        choices=[
            ("literature_review", "Literature Review"),
            ("abstract", "Abstract"),
            ("introduction", "Introduction"),
            ("methodology", "Methodology"),
            ("results", "Results"),
            ("discussion", "Discussion"),
            ("conclusion", "Conclusion"),
            ("citation", "Citation Formatting"),
        ],
    )
    content = models.TextField()
    improved_content = models.TextField(blank=True)

    # Assistance metadata
    suggestions = models.TextField(blank=True)  # JSON string
    style_improvements = models.TextField(blank=True)  # JSON string
    citation_corrections = models.TextField(blank=True)  # JSON string
    readability_score = models.FloatField(null=True, blank=True)
    word_count = models.IntegerField(default=0)

    # Generation settings
    tone = models.CharField(
        max_length=20,
        choices=[
            ("academic", "Academic"),
            ("formal", "Formal"),
            ("technical", "Technical"),
            ("accessible", "Accessible"),
        ],
        default="academic",
    )
    target_audience = models.CharField(max_length=50, default="researchers")

    def __str__(self):
        return f"Writing Assistance: {self.title} ({self.task_type})"

    class Meta:
        ordering = ["-created_at"]


class IntegrityCheck(BaseModel):
    """Model for academic integrity checks."""

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    content = models.TextField()

    # Check results
    plagiarism_score = models.FloatField(null=True, blank=True)
    similarity_matches = models.TextField(blank=True)  # JSON string
    citation_issues = models.TextField(blank=True)  # JSON string
    style_violations = models.TextField(blank=True)  # JSON string

    # Check metadata
    check_type = models.CharField(
        max_length=50,
        choices=[
            ("plagiarism", "Plagiarism Check"),
            ("citation", "Citation Verification"),
            ("style", "Style Check"),
            ("comprehensive", "Comprehensive Check"),
        ],
        default="comprehensive",
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
        default="pending",
    )

    # Results summary
    overall_score = models.FloatField(null=True, blank=True)
    recommendation = models.TextField(blank=True)
    issues_found = models.IntegerField(default=0)

    def __str__(self):
        return f"Integrity Check: {self.title}"

    class Meta:
        ordering = ["-created_at"]


class SearchHistory(BaseModel):
    """Model to track user search history."""

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    query = models.CharField(max_length=500)
    search_type = models.CharField(max_length=50, default="unified")
    sources = models.TextField(blank=True)  # JSON string
    filters = models.TextField(blank=True)  # JSON string
    results_count = models.IntegerField(default=0)
    execution_time = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Search: {self.query[:50]}... ({self.user.username})"

    class Meta:
        ordering = ["-created_at"]


class UserPreferences(BaseModel):
    """Model for user preferences and settings."""

    user = models.OneToOneField(User, on_delete=models.CASCADE)

    # Search preferences
    default_search_sources = models.TextField(
        default='["arxiv", "semantic_scholar"]'
    )  # JSON
    max_search_results = models.IntegerField(default=20)
    preferred_languages = models.TextField(default='["en"]')  # JSON

    # Notification preferences
    email_notifications = models.BooleanField(default=True)
    web_notifications = models.BooleanField(default=True)
    notification_frequency = models.CharField(max_length=20, default="weekly")

    # Content preferences
    preferred_podcast_style = models.CharField(max_length=50, default="summary")
    preferred_voice = models.CharField(max_length=50, default="alloy")
    auto_generate_summaries = models.BooleanField(default=True)

    # UI preferences
    theme = models.CharField(
        max_length=20,
        choices=[
            ("light", "Light"),
            ("dark", "Dark"),
            ("auto", "Auto"),
        ],
        default="light",
    )
    items_per_page = models.IntegerField(default=25)

    def __str__(self):
        return f"Preferences: {self.user.username}"


# Many-to-many relationship models
class PaperBookmark(BaseModel):
    """Model for user bookmarked papers."""

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE)
    notes = models.TextField(blank=True)
    tags = models.TextField(blank=True)  # JSON string

    class Meta:
        unique_together = ["user", "paper"]

    def __str__(self):
        return f"Bookmark: {self.paper.title[:50]}... ({self.user.username})"


class PaperRating(BaseModel):
    """Model for user ratings of papers."""

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    paper = models.ForeignKey(Paper, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])  # 1-5 stars
    review = models.TextField(blank=True)

    class Meta:
        unique_together = ["user", "paper"]

    def __str__(self):
        return f"Rating: {self.rating}/5 - {self.paper.title[:50]}..."


class ResearchProject(BaseModel):
    """Model for research projects."""

    title = models.CharField(max_length=200)
    description = models.TextField()
    keywords = models.TextField(blank=True)
    objectives = models.TextField(blank=True)
    owner = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="owned_projects"
    )
    collaborators = models.ManyToManyField(
        User, related_name="collaborated_projects", blank=True
    )

    # Project details
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
        default="pending",
    )

    # Results
    results = models.TextField(blank=True) # JSON string of results

    # Associated resources
    papers = models.ManyToManyField(Paper, blank=True)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)

    # Project metadata
    tags = models.TextField(blank=True)  # JSON string
    notes = models.TextField(blank=True)
    privacy = models.CharField(
        max_length=20,
        choices=[
            ("private", "Private"),
            ("collaborators", "Collaborators Only"),
            ("public", "Public"),
        ],
        default="private",
    )

    def __str__(self):
        return self.title

    class Meta:
        ordering = ["-created_at"]
