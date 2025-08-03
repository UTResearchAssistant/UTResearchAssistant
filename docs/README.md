# Documentation

This directory contains additional documentation for developers and users of
the AI research assistant.  You can add architecture diagrams, API
specifications, user guides and other materials here.

## Developer Guide

* **Backend:** The FastAPI backend is organised into layers (API,
  services, agents, models and database) to promote separation of
  concerns.  Refer to `backend/app/main.py` to see how routers are
  included and to the `services` package for business logic.
* **Frontend:** The React/Next.js front‑end lives under the `frontend/src`
  directory.  Pages define the routes of the application and
  components can be reused across pages.  API calls are centralised
  in `src/services/api.ts`.
* **Agents:** Agents encapsulate complex workflows that may involve
  multiple tools and model calls.  See `agents/summarizer_agent.py` for
  an example.
* **Services:** Background jobs for ingestion and monitoring run out of
  process.  These scripts can be executed on their own or scheduled
  via cron.  See `services/ingestion_service/` and `services/monitoring_service/`.

## User Guide

Once fully implemented, users will be able to:

1. **Search for papers:** Enter a query in the search bar on the home
   page and receive a list of matching papers.
2. **Summarise documents:** Click “Summarise” on a search result to
   request a concise summary along with citations.
3. **Track grants and conferences:** (Coming soon) Receive alerts for
   funding opportunities and relevant conferences based on your
   research interests.

This documentation is intentionally brief at this stage; it serves as
a placeholder until the full application is developed.