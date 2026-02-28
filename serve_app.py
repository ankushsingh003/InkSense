import http.server
import socketserver
import os
import sys

PORT = 3000
DIRECTORY = "ink-alchemist-web/dist"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Professional logging
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format%args))

if __name__ == "__main__":
    if not os.path.exists(DIRECTORY):
        print(f"Error: {DIRECTORY} not found. Please run 'npm run build' in ink-alchemist-web first.")
        sys.exit(1)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 Ink-Alchemist Server running at http://localhost:{PORT}")
        print(f"Serving files from: {os.path.abspath(DIRECTORY)}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()


