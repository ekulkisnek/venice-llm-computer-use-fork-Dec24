
# Adding Venice LLM API as proivder for Anthropic computer use demo

An interactive demo showcasing Claude's ability to control a computer through a Streamlit interface. This project allows Claude to perform various computer operations while providing a safe and controlled environment.

## Features

- Interactive chat interface with Claude
- Real-time computer control capabilities 
- Support for multiple AI providers (Anthropic, Bedrock, Vertex, Venice)
- HTTP exchange logging
- Screenshot capture and display
- Customizable system prompts
- Built-in security measures

## Getting Started

1. Click the "Run" button to start the application
2. Add your API key in the Secrets panel:
   - For Anthropic: Add `ANTHROPIC_API_KEY`
   - For other providers: Select provider from dropdown and add corresponding key

## Usage

1. The interface will open in the webview with two tabs:
   - Chat: Main interaction area with Claude
   - HTTP Exchange Logs: View API request/response details

2. Sidebar controls:
   - Provider selection (Anthropic, Bedrock, Vertex, Venice)
   - Model selection
   - Screenshot settings
   - Custom system prompt configuration
   - Reset button for environment refresh

3. Type commands in the chat input to interact with Claude

## Security Considerations

⚠️ **Important**: Computer use is a beta feature with unique risks:

- Never provide access to sensitive data or account credentials
- The model may follow commands found in web content even if they conflict with user instructions
- Consider having humans verify decisions with meaningful consequences
- Web content may override user instructions or cause unexpected behavior

## Environment

- Runs on Replit's infrastructure
- Uses Linux and Nix
- Firefox browser for web interactions
- Python environment with Streamlit interface

## Dependencies

- Python 3.10+
- Anthropic API client
- Streamlit
- Various system utilities

The environment is automatically configured when running on Replit.

## Contributing

Feel free to fork this project and make improvements. Use the Replit Projects feature for collaborative development.

## License

MIT License - See LICENSE file for details

Based on Anthropic's Computer Use Demo, adapted for Replit.
