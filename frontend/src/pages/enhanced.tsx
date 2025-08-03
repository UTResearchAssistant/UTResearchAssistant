import React, { useState, useEffect } from "react";
import Head from "next/head";
import {
  Search,
  Mic,
  Video,
  Bell,
  BookOpen,
  Users,
  TrendingUp,
  Download,
  Play,
  Pause,
  Settings,
  Globe,
  FileText,
  Shield,
  Zap,
} from "lucide-react";

// Enhanced API service
class EnhancedAPIService {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  // Enhanced search methods
  async enhancedSearch(query: string, options: any = {}) {
    const response = await fetch(`${this.baseUrl}/api/v2/search/enhanced`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        sources: options.sources || ["arxiv", "semantic_scholar", "pubmed"],
        max_results: options.maxResults || 20,
        filters: options.filters || {},
      }),
    });
    return await response.json();
  }

  async multilingualSearch(query: string, languages: string[]) {
    const response = await fetch(`${this.baseUrl}/api/v2/search/multilingual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        languages,
        max_results: 20,
      }),
    });
    return await response.json();
  }

  // Podcast methods
  async generatePodcast(paperIds: string[], options: any = {}) {
    const response = await fetch(`${this.baseUrl}/api/v2/podcast/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        paper_ids: paperIds,
        style: options.style || "conversational",
        duration_minutes: options.duration || 15,
        episode_type: options.type || "summary",
      }),
    });
    return await response.json();
  }

  // Video analysis methods
  async analyzeVideo(videoUrl: string, analysisType: string = "comprehensive") {
    const response = await fetch(`${this.baseUrl}/api/v2/video/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_url: videoUrl,
        analysis_type: analysisType,
      }),
    });
    return await response.json();
  }

  // Research alerts
  async createAlert(alertData: any) {
    const response = await fetch(`${this.baseUrl}/api/v2/alerts/create`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(alertData),
    });
    return await response.json();
  }

  async getAlerts() {
    const response = await fetch(`${this.baseUrl}/api/v2/alerts`);
    return await response.json();
  }

  // Writing assistance
  async analyzeWriting(text: string, assistanceType: string) {
    const response = await fetch(`${this.baseUrl}/api/v2/writing/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        assistance_type: assistanceType,
        context: "academic",
      }),
    });
    return await response.json();
  }

  // Integrity checking
  async checkIntegrity(text: string, checkType: string) {
    const response = await fetch(`${this.baseUrl}/api/v2/integrity/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        check_type: checkType,
      }),
    });
    return await response.json();
  }

  // Collaboration
  async findCollaborators(interests: string[]) {
    const response = await fetch(
      `${this.baseUrl}/api/v2/collaboration/find-researchers`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          research_interests: interests,
          max_results: 10,
        }),
      }
    );
    return await response.json();
  }

  // Analytics
  async getAnalytics() {
    const response = await fetch(`${this.baseUrl}/api/v2/analytics/usage`);
    return await response.json();
  }
}

const apiService = new EnhancedAPIService();

// Enhanced Search Component
const EnhancedSearchBar: React.FC<{
  onSearch: (query: string, options?: any) => void;
  loading?: boolean;
}> = ({ onSearch, loading = false }) => {
  const [query, setQuery] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [searchOptions, setSearchOptions] = useState({
    sources: ["arxiv", "semantic_scholar", "pubmed"],
    languages: ["en"],
    filters: {},
  });

  const handleSearch = () => {
    if (query.trim()) {
      onSearch(query, searchOptions);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
      <div className="flex items-center space-x-4 mb-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSearch()}
            placeholder="Search research papers across multiple databases..."
            className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <button
          onClick={handleSearch}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors disabled:opacity-50"
        >
          {loading ? "Searching..." : "Search"}
        </button>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-3 rounded-lg transition-colors"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>

      {showAdvanced && (
        <div className="border-t pt-4 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Data Sources
            </label>
            <div className="flex flex-wrap gap-2">
              {[
                "arxiv",
                "semantic_scholar",
                "pubmed",
                "google_scholar",
                "ieee",
              ].map((source) => (
                <label key={source} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={searchOptions.sources.includes(source)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSearchOptions((prev) => ({
                          ...prev,
                          sources: [...prev.sources, source],
                        }));
                      } else {
                        setSearchOptions((prev) => ({
                          ...prev,
                          sources: prev.sources.filter((s) => s !== source),
                        }));
                      }
                    }}
                    className="mr-2"
                  />
                  <span className="text-sm capitalize">
                    {source.replace("_", " ")}
                  </span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Languages
            </label>
            <div className="flex flex-wrap gap-2">
              {[
                { code: "en", name: "English" },
                { code: "es", name: "Spanish" },
                { code: "fr", name: "French" },
                { code: "de", name: "German" },
                { code: "zh", name: "Chinese" },
                { code: "ja", name: "Japanese" },
              ].map((lang) => (
                <label key={lang.code} className="flex items-center">
                  <input
                    type="checkbox"
                    checked={searchOptions.languages.includes(lang.code)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSearchOptions((prev) => ({
                          ...prev,
                          languages: [...prev.languages, lang.code],
                        }));
                      } else {
                        setSearchOptions((prev) => ({
                          ...prev,
                          languages: prev.languages.filter(
                            (l) => l !== lang.code
                          ),
                        }));
                      }
                    }}
                    className="mr-2"
                  />
                  <span className="text-sm">{lang.name}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Podcast Generation Component
const PodcastGenerator: React.FC<{
  selectedPapers: any[];
}> = ({ selectedPapers }) => {
  const [podcastOptions, setPodcastOptions] = useState({
    style: "conversational",
    duration: 15,
    type: "summary",
  });
  const [generating, setGenerating] = useState(false);
  const [generatedPodcast, setGeneratedPodcast] = useState(null);

  const generatePodcast = async () => {
    if (selectedPapers.length === 0) {
      alert("Please select at least one paper");
      return;
    }

    setGenerating(true);
    try {
      const result = await apiService.generatePodcast(
        selectedPapers.map((p) => p.id),
        podcastOptions
      );
      setGeneratedPodcast(result.episode);
    } catch (error) {
      console.error("Error generating podcast:", error);
      alert("Error generating podcast");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Mic className="w-6 h-6 text-purple-600" />
        <h3 className="text-xl font-semibold">Generate Research Podcast</h3>
      </div>

      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Podcast Style
          </label>
          <select
            value={podcastOptions.style}
            onChange={(e) =>
              setPodcastOptions((prev) => ({ ...prev, style: e.target.value }))
            }
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
          >
            <option value="conversational">Conversational</option>
            <option value="academic">Academic</option>
            <option value="narrative">Narrative</option>
            <option value="educational">Educational</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Episode Type
          </label>
          <select
            value={podcastOptions.type}
            onChange={(e) =>
              setPodcastOptions((prev) => ({ ...prev, type: e.target.value }))
            }
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
          >
            <option value="summary">Summary</option>
            <option value="interview">Interview</option>
            <option value="debate">Debate</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Duration (minutes)
          </label>
          <input
            type="number"
            value={podcastOptions.duration}
            onChange={(e) =>
              setPodcastOptions((prev) => ({
                ...prev,
                duration: parseInt(e.target.value),
              }))
            }
            min="5"
            max="60"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
          />
        </div>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-600">
          {selectedPapers.length} papers selected
        </span>
        <button
          onClick={generatePodcast}
          disabled={generating || selectedPapers.length === 0}
          className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-2 rounded-lg transition-colors disabled:opacity-50"
        >
          {generating ? "Generating..." : "Generate Podcast"}
        </button>
      </div>

      {generatedPodcast && (
        <div className="mt-6 p-4 bg-purple-50 rounded-lg">
          <h4 className="font-semibold text-purple-900 mb-2">
            {generatedPodcast.title}
          </h4>
          <p className="text-purple-700 text-sm mb-3">
            {generatedPodcast.description}
          </p>
          <div className="flex items-center space-x-4">
            <button className="flex items-center space-x-2 bg-purple-600 text-white px-4 py-2 rounded-lg">
              <Play className="w-4 h-4" />
              <span>Play</span>
            </button>
            <button className="flex items-center space-x-2 bg-gray-600 text-white px-4 py-2 rounded-lg">
              <Download className="w-4 h-4" />
              <span>Download</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// Video Analysis Component
const VideoAnalyzer: React.FC = () => {
  const [videoUrl, setVideoUrl] = useState("");
  const [analysisType, setAnalysisType] = useState("comprehensive");
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);

  const analyzeVideo = async () => {
    if (!videoUrl.trim()) {
      alert("Please enter a video URL");
      return;
    }

    setAnalyzing(true);
    try {
      const result = await apiService.analyzeVideo(videoUrl, analysisType);
      setAnalysis(result.analysis);
    } catch (error) {
      console.error("Error analyzing video:", error);
      alert("Error analyzing video");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Video className="w-6 h-6 text-red-600" />
        <h3 className="text-xl font-semibold">Video Analysis</h3>
      </div>

      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Video URL
          </label>
          <input
            type="url"
            value={videoUrl}
            onChange={(e) => setVideoUrl(e.target.value)}
            placeholder="https://youtube.com/watch?v=..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Analysis Type
          </label>
          <select
            value={analysisType}
            onChange={(e) => setAnalysisType(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500"
          >
            <option value="comprehensive">Comprehensive</option>
            <option value="lecture">Lecture</option>
            <option value="presentation">Conference Presentation</option>
          </select>
        </div>
      </div>

      <button
        onClick={analyzeVideo}
        disabled={analyzing}
        className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg transition-colors disabled:opacity-50 mb-6"
      >
        {analyzing ? "Analyzing..." : "Analyze Video"}
      </button>

      {analysis && (
        <div className="space-y-4">
          <div className="p-4 bg-red-50 rounded-lg">
            <h4 className="font-semibold text-red-900 mb-2">
              {analysis.title}
            </h4>
            <p className="text-red-700 text-sm">{analysis.summary}</p>
          </div>

          {analysis.key_points && analysis.key_points.length > 0 && (
            <div>
              <h5 className="font-medium text-gray-900 mb-2">Key Points</h5>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                {analysis.key_points.map((point, index) => (
                  <li key={index}>{point}</li>
                ))}
              </ul>
            </div>
          )}

          {analysis.topics && analysis.topics.length > 0 && (
            <div>
              <h5 className="font-medium text-gray-900 mb-2">Topics</h5>
              <div className="flex flex-wrap gap-2">
                {analysis.topics.map((topic, index) => (
                  <span
                    key={index}
                    className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs"
                  >
                    {topic}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Research Alerts Component
const ResearchAlerts: React.FC = () => {
  const [alerts, setAlerts] = useState([]);
  const [newAlert, setNewAlert] = useState({
    alert_type: "keyword",
    query: "",
    frequency: "weekly",
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadAlerts();
  }, []);

  const loadAlerts = async () => {
    try {
      const result = await apiService.getAlerts();
      setAlerts(result.alerts || []);
    } catch (error) {
      console.error("Error loading alerts:", error);
    }
  };

  const createAlert = async () => {
    if (!newAlert.query.trim()) {
      alert("Please enter a search query");
      return;
    }

    setLoading(true);
    try {
      await apiService.createAlert(newAlert);
      setNewAlert({ alert_type: "keyword", query: "", frequency: "weekly" });
      loadAlerts();
    } catch (error) {
      console.error("Error creating alert:", error);
      alert("Error creating alert");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Bell className="w-6 h-6 text-yellow-600" />
        <h3 className="text-xl font-semibold">Research Alerts</h3>
      </div>

      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Alert Type
          </label>
          <select
            value={newAlert.alert_type}
            onChange={(e) =>
              setNewAlert((prev) => ({ ...prev, alert_type: e.target.value }))
            }
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-yellow-500"
          >
            <option value="keyword">Keyword</option>
            <option value="author">Author</option>
            <option value="journal">Journal</option>
            <option value="topic">Topic</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Search Query
          </label>
          <input
            type="text"
            value={newAlert.query}
            onChange={(e) =>
              setNewAlert((prev) => ({ ...prev, query: e.target.value }))
            }
            placeholder="Enter search terms..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-yellow-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Frequency
          </label>
          <select
            value={newAlert.frequency}
            onChange={(e) =>
              setNewAlert((prev) => ({ ...prev, frequency: e.target.value }))
            }
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-yellow-500"
          >
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </div>
      </div>

      <button
        onClick={createAlert}
        disabled={loading}
        className="bg-yellow-600 hover:bg-yellow-700 text-white px-6 py-2 rounded-lg transition-colors disabled:opacity-50 mb-6"
      >
        {loading ? "Creating..." : "Create Alert"}
      </button>

      {alerts.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-4">Active Alerts</h4>
          <div className="space-y-3">
            {alerts.map((alert, index) => (
              <div key={index} className="p-3 bg-yellow-50 rounded-lg border">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="font-medium text-yellow-900">
                      {alert.query}
                    </span>
                    <div className="text-sm text-yellow-700">
                      {alert.alert_type} • {alert.frequency}
                    </div>
                  </div>
                  <button className="text-yellow-600 hover:text-yellow-800">
                    <span className="text-sm">Delete</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Writing Assistant Component
const WritingAssistant: React.FC = () => {
  const [text, setText] = useState("");
  const [assistanceType, setAssistanceType] = useState("grammar");
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);

  const analyzeText = async () => {
    if (!text.trim()) {
      alert("Please enter some text to analyze");
      return;
    }

    setAnalyzing(true);
    try {
      const result = await apiService.analyzeWriting(text, assistanceType);
      setAnalysis(result.analysis);
    } catch (error) {
      console.error("Error analyzing writing:", error);
      alert("Error analyzing writing");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center space-x-2 mb-4">
        <FileText className="w-6 h-6 text-green-600" />
        <h3 className="text-xl font-semibold">Writing Assistant</h3>
      </div>

      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Assistance Type
          </label>
          <select
            value={assistanceType}
            onChange={(e) => setAssistanceType(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
          >
            <option value="grammar">Grammar</option>
            <option value="style">Style</option>
            <option value="clarity">Clarity</option>
            <option value="structure">Structure</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Text to Analyze
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste your academic text here..."
            rows={8}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500"
          />
        </div>
      </div>

      <button
        onClick={analyzeText}
        disabled={analyzing}
        className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg transition-colors disabled:opacity-50 mb-6"
      >
        {analyzing ? "Analyzing..." : "Analyze Writing"}
      </button>

      {analysis && (
        <div className="space-y-6">
          {analysis.suggestions && analysis.suggestions.length > 0 && (
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Suggestions</h4>
              <div className="space-y-2">
                {analysis.suggestions.map((suggestion, index) => (
                  <div
                    key={index}
                    className="p-3 bg-green-50 rounded-lg border-l-4 border-green-400"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <span className="font-medium text-green-900">
                          {suggestion.type}
                        </span>
                        <p className="text-green-700 text-sm mt-1">
                          {suggestion.message}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {analysis.metrics && (
            <div>
              <h4 className="font-medium text-gray-900 mb-3">
                Readability Metrics
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-3 bg-gray-50 rounded-lg text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {analysis.metrics.readability_score}
                  </div>
                  <div className="text-sm text-gray-600">Readability Score</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {analysis.metrics.word_count}
                  </div>
                  <div className="text-sm text-gray-600">Words</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg text-center">
                  <div className="text-2xl font-bold text-gray-900">
                    {analysis.metrics.sentence_count}
                  </div>
                  <div className="text-sm text-gray-600">Sentences</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg text-center">
                  <div className="text-xl font-bold text-gray-900">
                    {analysis.metrics.grade_level}
                  </div>
                  <div className="text-sm text-gray-600">Grade Level</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Main Enhanced Research Assistant Component
const EnhancedResearchAssistant: React.FC = () => {
  const [searchResults, setSearchResults] = useState([]);
  const [selectedPapers, setSelectedPapers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("search");

  const handleSearch = async (query: string, options?: any) => {
    setLoading(true);
    try {
      let result;
      if (options?.languages && options.languages.length > 1) {
        result = await apiService.multilingualSearch(query, options.languages);
      } else {
        result = await apiService.enhancedSearch(query, options);
      }
      setSearchResults(result.results || []);
    } catch (error) {
      console.error("Search error:", error);
      alert("Search failed");
    } finally {
      setLoading(false);
    }
  };

  const togglePaperSelection = (paper: any) => {
    setSelectedPapers((prev) => {
      const isSelected = prev.some((p) => p.id === paper.id);
      if (isSelected) {
        return prev.filter((p) => p.id !== paper.id);
      } else {
        return [...prev, paper];
      }
    });
  };

  const tabs = [
    { id: "search", label: "Enhanced Search", icon: Search },
    { id: "podcast", label: "Podcast Generator", icon: Mic },
    { id: "video", label: "Video Analysis", icon: Video },
    { id: "alerts", label: "Research Alerts", icon: Bell },
    { id: "writing", label: "Writing Assistant", icon: FileText },
    { id: "collaboration", label: "Collaboration", icon: Users },
  ];

  return (
    <>
      <Head>
        <title>Enhanced AI Research Assistant</title>
        <meta
          name="description"
          content="Comprehensive AI-powered research assistant with podcasts, video analysis, and advanced features"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Zap className="w-8 h-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">
                  Enhanced Research Assistant
                </h1>
              </div>
              <div className="flex items-center space-x-4">
                <span className="text-sm text-gray-600">
                  v2.0 - Advanced Features
                </span>
                <div className="flex space-x-2">
                  <Globe
                    className="w-5 h-5 text-green-500"
                    title="Multilingual Support"
                  />
                  <Shield
                    className="w-5 h-5 text-blue-500"
                    title="Integrity Checking"
                  />
                  <Mic
                    className="w-5 h-5 text-purple-500"
                    title="Podcast Generation"
                  />
                  <Video
                    className="w-5 h-5 text-red-500"
                    title="Video Analysis"
                  />
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Tab Navigation */}
        <nav className="bg-white border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex space-x-8 overflow-x-auto">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                      activeTab === tab.id
                        ? "border-blue-500 text-blue-600"
                        : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {activeTab === "search" && (
            <div className="space-y-6">
              <EnhancedSearchBar onSearch={handleSearch} loading={loading} />

              {searchResults.length > 0 && (
                <div className="bg-white rounded-lg shadow-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold">
                      Search Results ({searchResults.length})
                    </h3>
                    <span className="text-sm text-gray-600">
                      {selectedPapers.length} selected for podcast generation
                    </span>
                  </div>

                  <div className="space-y-4">
                    {searchResults.map((paper, index) => (
                      <div key={index} className="border rounded-lg p-4">
                        <div className="flex items-start space-x-4">
                          <input
                            type="checkbox"
                            checked={selectedPapers.some(
                              (p) => p.id === paper.id
                            )}
                            onChange={() => togglePaperSelection(paper)}
                            className="mt-1"
                          />
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 mb-2">
                              {paper.title}
                            </h4>
                            <p className="text-gray-700 text-sm mb-2">
                              {paper.authors?.join(", ")} • {paper.source} •
                              Citations: {paper.citation_count || 0}
                            </p>
                            <p className="text-gray-600 text-sm">
                              {paper.abstract?.substring(0, 200)}...
                            </p>
                            <div className="mt-2 flex items-center space-x-4">
                              {paper.url && (
                                <a
                                  href={paper.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:text-blue-800 text-sm"
                                >
                                  View Paper
                                </a>
                              )}
                              {paper.pdf_url && (
                                <a
                                  href={paper.pdf_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-green-600 hover:text-green-800 text-sm"
                                >
                                  PDF
                                </a>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "podcast" && (
            <PodcastGenerator selectedPapers={selectedPapers} />
          )}

          {activeTab === "video" && <VideoAnalyzer />}

          {activeTab === "alerts" && <ResearchAlerts />}

          {activeTab === "writing" && <WritingAssistant />}

          {activeTab === "collaboration" && (
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center space-x-2 mb-4">
                <Users className="w-6 h-6 text-indigo-600" />
                <h3 className="text-xl font-semibold">
                  Research Collaboration
                </h3>
              </div>
              <div className="text-center py-12">
                <Users className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <h4 className="text-lg font-medium text-gray-900 mb-2">
                  Find Research Collaborators
                </h4>
                <p className="text-gray-600 mb-6">
                  Connect with researchers who share your interests and expand
                  your research network.
                </p>
                <button className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-lg transition-colors">
                  Find Collaborators
                </button>
              </div>
            </div>
          )}
        </main>
      </div>
    </>
  );
};

export default EnhancedResearchAssistant;
