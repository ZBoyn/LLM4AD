from ...base import LLM, Function, Program, TextFunctionProgramConverter, SampleTrimmer
import re
import ast # Import the Abstract Syntax Tree module


class EvoMCTSSampler:

    def __init__(self, llm: LLM, template_program_str: str):
        self.llm = llm
        self.template_program_str = template_program_str

    def _is_syntactically_valid(self, code: str) -> bool:
        """Checks if the given code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True
        except (SyntaxError, ValueError): # Catch more errors
            return False

    def get_thought_and_function(self, task_description, prompt):
        response = self.llm.draw_sample(prompt)
        
        # --- CRITICAL FIX: Use the SampleTrimmer for robust parsing ---
        thought = self._trim_thought_from_response(response)
        code = SampleTrimmer.trim_preface_of_function(response)
        function = SampleTrimmer.sample_to_function(code, self.template_program_str)
        
        if function is None:
            print("    Sampler failed to extract a valid function from the response.")
            return None, None
            
        if thought is None:
            thought = "An algorithm generated based on the provided template."

        return thought, function

    def _trim_thought_from_response(self, response: str) -> str | None:
        """Extracts the thought from the response, assuming it's in {curly braces}."""
        try:
            # This regex is more robust for multiline thoughts
            pattern = r'\{(.*?)\}' 
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
            return None
        except Exception:
            return None

    def get_reflection(self, prompt: str) -> str | None:
        """Invokes the LLM for a reflection and returns the raw text response."""
        try:
            response = self.llm.draw_sample(prompt)
            # For reflection, we often want the whole cleaned response,
            # not just a thought in braces.
            # A simple cleanup could be to strip whitespace.
            return response.strip() if response else None
        except Exception:
            return None