# z3_solver.py

import z3
import logging
import argparse
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# Configure logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/z3_server.log")
file_handler.setLevel(logging.WARNING)  # Log both warnings and errors to file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[console_handler, file_handler]
)

logger = logging.getLogger(__name__)


class Z3Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle health check requests"""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        """Handle SMT formula evaluation requests"""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"No content provided")
                return

            post_data = self.rfile.read(content_length)
            formula = post_data.decode("utf-8")

            logger.info(f"Received formula: {formula[:100]}...")  # Log first 100 chars

            result = self.evaluate_formula(formula)

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(result.encode("utf-8"))

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            logger.error(traceback.format_exc())
            self.send_response(500)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode("utf-8"))

    def evaluate_formula(self, formula):
        try:
            # create a new solver for each request
            solver = z3.Solver()

            # parse the SMT-LIB formula
            parsed_formula = z3.parse_smt2_string(formula)

            # add the formula to the solver
            solver.add(parsed_formula)

            # check satisfiability
            result = solver.check()

            if result == z3.sat:
                return "sat"
            elif result == z3.unsat:
                return "unsat"
            else:
                return "unknown"

        except Exception as e:
            logger.error(f"Error evaluating formula: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def log_message(self, format, *args):
        """Override to use our logger instead of stderr"""
        logger.info("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))


def run_server(port):
    server_address = ("localhost", port)
    httpd = HTTPServer(server_address, Z3Handler)
    logger.info(f"Starting Z3 server on port {port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        httpd.server_close()
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z3 SMT Solver Server")
    parser.add_argument("--port", type=int, default=8000, help="Port number to run the server on")
    args = parser.parse_args()

    run_server(args.port)
