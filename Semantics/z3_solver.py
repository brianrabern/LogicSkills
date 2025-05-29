# z3_solver.py

import z3
import logging
import argparse
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import gc
import re

# Constants
Z3_TIMEOUT = 5000  # 5 second timeout in milliseconds

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
            print("posting!!")
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"No content provided")
                return

            post_data = self.rfile.read(content_length)

            # Check if this is a model request
            if self.path == "/model":
                print("Getting model....")
                result = self.get_model(json.loads(post_data))
            else:
                print("Evaluating formula....")
                result = self.evaluate_formula(post_data.decode("utf-8"))

            try:
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(result.encode("utf-8"))
            except BrokenPipeError:
                logger.warning("Client disconnected before response could be sent")
                return
            except Exception as e:
                logger.error(f"Error sending response: {str(e)}")
                return

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            logger.error(traceback.format_exc())
            try:
                self.send_response(500)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Error: {str(e)}".encode("utf-8"))
            except BrokenPipeError:
                logger.warning("Client disconnected before error response could be sent")
                return
            except Exception as e:
                logger.error(f"Error sending error response: {str(e)}")
                return

    def evaluate_formula(self, formula):
        try:
            # Create a fresh Z3 context to isolate memory
            ctx = z3.Context()
            solver = z3.Solver(ctx=ctx)
            solver.set("timeout", Z3_TIMEOUT)

            try:
                parsed_formula = z3.parse_smt2_string(formula, ctx=ctx)
            except Exception as e:
                logger.error(f"Error parsing formula: {str(e)}")
                return "error: invalid formula"

            try:
                solver.add(parsed_formula)
            except Exception as e:
                logger.error(f"Error adding formula to solver: {str(e)}")

                return "error: invalid formula"

            try:
                result = solver.check()
            except Exception as e:
                logger.error(f"Error checking satisfiability: {str(e)}")
                return "error: solver error"

            if result == z3.sat:
                response = "sat"
            elif result == z3.unsat:
                response = "unsat"
            else:
                response = "unknown"

            # Explicit cleanup
            del solver
            del parsed_formula
            del ctx
            gc.collect()  # Force full cleanup of memory

            return response

        except Exception as e:
            logger.error(f"Error evaluating formula: {str(e)}")
            logger.error(traceback.format_exc())
            return "error: internal error"

    def serialize_model(self, model_dict):
        def extract_val_id(val):
            match = re.search(r"!val!(\d+)", str(val))
            return int(match.group(1)) if match else str(val)

        result = {}

        for k, v in model_dict.items():
            if isinstance(v, list):
                processed = []
                for item in v:
                    if isinstance(item, tuple):
                        processed.append(tuple(extract_val_id(x) for x in item))
                    else:
                        processed.append(extract_val_id(item))
                result[k] = processed
            else:
                result[k] = extract_val_id(v)

        return json.dumps(result)

    def get_model(self, data):
        """Get the model for a satisfiable formula"""
        try:
            logger.debug("Starting get_model...")
            # Create a fresh Z3 context to isolate memory
            ctx = z3.Context()
            solver = z3.Solver(ctx=ctx)
            solver.set("timeout", Z3_TIMEOUT)
            formula = data["smt2"]

            try:
                logger.debug(f"Parsing formula: {formula}")
                parsed_formula = z3.parse_smt2_string(formula, ctx=ctx)
                logger.debug(f"Parsed formula: {parsed_formula}")
            except Exception as e:
                logger.error(f"Error parsing formula: {str(e)}")
                return "error: invalid formula"
            try:
                logger.debug("Adding formula to solver...")
                solver.add(parsed_formula)
            except Exception as e:
                logger.error(f"Error adding formula to solver: {str(e)}")
                return "error: invalid formula"
            try:
                logger.debug("Checking satisfiability...")
                result = solver.check()
                logger.debug(f"Solver result: {result}")
            except Exception as e:
                logger.error(f"Error checking satisfiability: {str(e)}")
                return "error: solver error"

            if result == z3.sat:
                model = solver.model()
                logger.debug(f"Raw model: {model}")

                Object = z3.DeclareSort("Object", ctx=ctx)
                domain_elems = model.get_universe(Object)
                logger.debug(f"Domain elements: {domain_elems}")

                model_dict = {}

                for d in model.decls():
                    interp = model.get_interp(d)
                    logger.debug(f"model get_interp: {interp}")

                    if isinstance(interp, z3.FuncInterp):
                        arity = interp.arity()

                        if arity == 1:
                            true_elements = [
                                e for e in domain_elems if z3.is_true(model.evaluate(d(e), model_completion=True))
                            ]
                            model_dict[d.name()] = true_elements

                        elif arity == 2:
                            true_pairs = [
                                (x, y)
                                for x in domain_elems
                                for y in domain_elems
                                if z3.is_true(model.evaluate(d(x, y), model_completion=True))
                            ]
                            model_dict[d.name()] = true_pairs

                    else:
                        model_dict[d.name()] = interp

                # handle declared names and predicate whose interpretation is arbitrary
                names = data["names"]
                predicates = data["monadic_predicates"] + data["binary_predicates"]
                for name in names:
                    if name not in model_dict:
                        model_dict[name] = domain_elems[0]
                for predicate in predicates:
                    if predicate not in model_dict:
                        model_dict[predicate] = []
                model_dict["domain"] = [x for x in domain_elems]
                logger.debug(f"Model interpretation: {model_dict}")
                response = self.serialize_model(model_dict)

            else:
                response = None
            # Explicit cleanup
            del solver
            del parsed_formula
            del ctx
            gc.collect()  # Force full cleanup of memory

            return response

        except Exception as e:
            logger.error(f"Error in get_model: {str(e)}")
            logger.error(traceback.format_exc())
            return "error: internal error"

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
    parser.add_argument("--port", type=int, default=8001, help="Port number to run the server on")
    args = parser.parse_args()

    run_server(args.port)
