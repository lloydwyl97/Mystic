"""
Backend App Module - New Clean Architecture

This module implements the new clean architecture for the Mystic Trading Platform.
Follows clean architecture principles with separation of concerns:
- Core: Configuration, logging, lifecycle
- Domain: Models and repositories
- Adapters: Database, HTTP, cache interfaces
- Services: Business logic located in `backend/services/` (active, in use)
- API: HTTP endpoints and routing

All components connect to live backend on port 8000.
All data operations use live endpoints - no fallback/hardcoded data.

Note: Services are located in `backend/services/` and actively used throughout the codebase.
"""
