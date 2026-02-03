import requests
import logging
from Syntax.convert_to_smt import ast_to_smt2
from Utils.helpers import ast_from_json
from Utils.logging_config import setup_logging
import traceback

# set up logging for this module
log_file = setup_logging("z3_evaluator")
logger = logging.getLogger(__name__)

# configuration
Z3_SERVER = "http://localhost:8001"  # single z3 server
logger.info(f"Using Z3 server at: {Z3_SERVER}")


def evaluate(sentence_ast, convert_json=False):
    """
    Evaluate a sentence AST using Z3.
    """
    try:
        if convert_json:
            sentence_ast = ast_from_json(sentence_ast)
        logger.info(f"Evaluating sentence AST: {sentence_ast}")

        # convert to smt format
        data = ast_to_smt2(sentence_ast)["smt2"]
        logger.info(f"SMT sent to Z3: {data}")

        endpoint = Z3_SERVER

        # send to z3 server
        response = requests.post(endpoint, data=data, headers={"Content-Type": "text/plain"}, timeout=10)
        response.raise_for_status()

        result = response.text.strip()
        logger.info(f"Z3 response: {result}")
        return result

    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Z3 server: {str(e)}")
        return "error"
    except Exception as e:
        error_msg = f"Unexpected error in evaluate function. AST: {sentence_ast}\nError: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return "error"
