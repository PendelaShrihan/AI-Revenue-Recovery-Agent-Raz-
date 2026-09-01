# Project Security Rules

## Environment Variables
- NEVER read, print, log, or expose any value from the `.env` file
- NEVER hardcode any API key, secret, password, or credential anywhere in the code
- ALWAYS use `os.getenv("KEY_NAME")` to read env vars and use them directly
- NEVER print or return the actual value of any env var in any output, log, or test result
- If an env var is missing, raise a clear error like `"GEMINI_API_KEY is not set"` — never expose what the value should be

## Git Rules
- `.env` must always be in `.gitignore` — never stage or commit it
- Only `.env.example` with placeholder values is allowed in the repo
- Before every commit run `git status` and confirm `.env` is not staged

## API Rules
- Always use real APIs — never mock or stub external calls in production code
- If a real API call fails, print the error and stop — never silently fall back to fake data
- All API keys are read from `.env` only

## Code Rules
- Never print API keys, secrets, or tokens anywhere including debug logs
- Never log request headers that contain authorization tokens
- Never store sensitive data in plain text anywhere in the codebase
