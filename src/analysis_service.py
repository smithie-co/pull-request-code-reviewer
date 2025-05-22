import logging
import json
from typing import Optional, List, Dict, Any

from src.bedrock_handler import BedrockHandler
from src import config

logger = logging.getLogger(__name__)

class AnalysisService:
    """Orchestrates code analysis using Bedrock models via BedrockHandler."""

    def __init__(self, bedrock_handler: BedrockHandler):
        """
        Initializes the AnalysisService.

        Args:
            bedrock_handler: An instance of BedrockHandler.
        """
        self.bedrock_handler = bedrock_handler

    def analyze_code_changes(self, diff_content: str, 
                               model_id_override: Optional[str] = None) -> str:
        """
        Analyzes the given diff content using the HEAVY_MODEL for in-depth review.

        Args:
            diff_content: The code diff to analyze.
            model_id_override: Optional Bedrock model ID to override the default HEAVY_MODEL_ID.

        Returns:
            The analysis result from the heavy model.
        
        Raises:
            ValueError: If diff_content is empty or model ID is not configured.
            RuntimeError: If model invocation fails.
        """
        if not diff_content:
            raise ValueError("Diff content cannot be empty for analysis.")
        
        heavy_model_id = model_id_override or config.HEAVY_MODEL_ID
        if not heavy_model_id:
            raise ValueError("HEAVY_MODEL_ID is not configured for code analysis.")

        # Basic prompt, can be significantly improved with more context and specific instructions.
        prompt = f"""Human: You are an expert code reviewer. Please analyze the following code diff for potential bugs, 
style issues, security vulnerabilities, and areas for improvement. Provide specific feedback. 
Only analyze the provided diff, do not make assumptions about code outside this diff.

<diff>
{diff_content}
</diff>

Assistant: Analysis results:"""
        
        logger.info(f"Analyzing code changes using model: {heavy_model_id}")
        try:
            analysis = self.bedrock_handler.invoke_model(
                model_id=heavy_model_id,
                prompt=prompt,
                # Consider adjusting max_tokens, temperature for heavy analysis
                max_tokens=3072, # Increased for potentially longer analysis
                temperature=0.5  # Lower for more deterministic/factual analysis
            )
            return analysis
        except Exception as e:
            logger.error(f"Error during heavy model analysis (model: {heavy_model_id}): {e}")
            raise RuntimeError(f"Heavy model analysis failed: {e}") from e

    def analyze_heavy_model_output(self, heavy_model_output: str, 
                                     diff_content: str,
                                     model_id_override: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyzes the output from the HEAVY_MODEL using DEEPSEEK_MODEL to extract structured, 
        actionable suggestions with file paths and line numbers.

        Args:
            heavy_model_output: The output from the heavy model's analysis.
            diff_content: The original diff content, provided in case the LLM needs to refer to it
                          to accurately determine file paths and line numbers.
            model_id_override: Optional Bedrock model ID to override the default DEEPSEEK_MODEL_ID.

        Returns:
            A list of dictionaries, where each dictionary represents a suggestion and contains:
            {'file_path': str, 'line': int, 'suggestion': str}.
            Returns an empty list if no actionable suggestions are found or in case of errors.

        Raises:
            ValueError: If heavy_model_output is empty or model ID is not configured.
            RuntimeError: If model invocation fails or output parsing is unsuccessful.
        """
        if not heavy_model_output:
            logger.warning("Heavy model output is empty. Cannot perform refined analysis.")
            return []

        deepseek_model_id = model_id_override or config.DEEPSEEK_MODEL_ID
        if not deepseek_model_id:
            logger.error("DEEPSEEK_MODEL_ID is not configured for refining analysis.")
            raise ValueError("DEEPSEEK_MODEL_ID is not configured for refining analysis.")

        prompt = f"""Human: You are an AI assistant that extracts actionable code review suggestions from an analysis. 
Based on the original code diff and the provided analysis, identify specific, actionable suggestions. 
For each suggestion, determine the file path and the relevant line number within that file as shown in the diff or the new version of the file.

Original Code Diff:
<diff>
{diff_content}
</diff>

Analysis from another AI:
<previous_analysis>
{heavy_model_output}
</previous_analysis>

Please format your output as a JSON array, where each object in the array has the following keys: "file_path" (string), "line" (integer, the specific line number your suggestion pertains to in the changed file), and "suggestion" (string, your concise suggestion for that line).
If no specific, actionable, line-level suggestions can be extracted, return an empty JSON array [].

Example of desired JSON output format:
[
  {{
    "file_path": "src/example.py",
    "line": 42,
    "suggestion": "Consider using a more descriptive variable name instead of 'x'."
  }},
  {{
    "file_path": "tests/test_utils.py",
    "line": 15,
    "suggestion": "Add a test case for empty input."
  }}
]

Assistant:JSON Output:"""

        logger.info(f"Extracting structured suggestions using model: {deepseek_model_id}")
        try:
            raw_structured_output = self.bedrock_handler.invoke_model(
                model_id=deepseek_model_id,
                prompt=prompt,
                max_tokens=2048, 
                temperature=0.3
            )
            
            logger.debug(f"Raw structured output from model {deepseek_model_id}: {raw_structured_output}")
            
            # Attempt to extract JSON, handling potential markdown code blocks
            json_str_to_parse = raw_structured_output.strip()

            if json_str_to_parse.startswith("```json"):
                json_str_to_parse = json_str_to_parse[len("```json"):].strip()
                if json_str_to_parse.endswith("```"):
                    json_str_to_parse = json_str_to_parse[:-len("```")].strip()
            elif json_str_to_parse.startswith("```"):
                json_str_to_parse = json_str_to_parse[len("```"):].strip()
                if json_str_to_parse.endswith("```"):
                    json_str_to_parse = json_str_to_parse[:-len("```")].strip()

            # Ensure it looks like an array or object before trying to parse directly
            # (though LLMs might just return the array without a full block sometimes)
            # The main attempt is to clean the string and then try parsing.

            try:
                # Try to parse the cleaned output
                parsed_suggestions = json.loads(json_str_to_parse)
            except json.JSONDecodeError:
                # If parsing the cleaned output fails, try to find a JSON array within the text
                logger.warning(f"Initial JSON parse of cleaned output failed. Trying to find array in: {json_str_to_parse[:200]}...") # Log snippet
                json_start_index = json_str_to_parse.find('[')
                json_end_index = json_str_to_parse.rfind(']')
                
                if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                    extracted_json_str = json_str_to_parse[json_start_index : json_end_index+1]
                    logger.debug(f"Attempting to parse extracted substring: {extracted_json_str[:200]}...")
                    try:
                        parsed_suggestions = json.loads(extracted_json_str)
                    except json.JSONDecodeError as e_inner:
                        logger.error(f"Failed to decode extracted JSON array. Error: {e_inner}. Extracted substring: {extracted_json_str[:500]}...")
                        logger.error(f"Original raw output (first 500 chars) was: {raw_structured_output[:500]}...")
                        return [] 
                else:
                    logger.error(f"Could not find valid JSON array delimiters '[' and ']' in the model output after initial parse failed. Cleaned output was: {json_str_to_parse[:500]}...")
                    logger.error(f"Original raw output (first 500 chars) was: {raw_structured_output[:500]}...")
                    return []

            if not isinstance(parsed_suggestions, list):
                logger.warning(f"Parsed JSON output is not a list as expected. Output: {parsed_suggestions}")
                return []

            validated_suggestions: List[Dict[str, Any]] = []
            for item in parsed_suggestions:
                if isinstance(item, dict) and \
                   all(k in item for k in ["file_path", "line", "suggestion"]) and \
                   isinstance(item.get("file_path"), str) and \
                   isinstance(item.get("line"), int) and \
                   isinstance(item.get("suggestion"), str):
                    validated_suggestions.append(item)
                else:
                    logger.warning(f"Skipping malformed suggestion item from LLM: {item}")
            
            logger.info(f"Successfully extracted {len(validated_suggestions)} structured suggestions.")
            return validated_suggestions

        except Exception as e:
            logger.error(f"Error during structured suggestion extraction (model: {deepseek_model_id}): {e}", exc_info=True)
            return []

    def summarize_changes(self, diff_content: str, 
                            model_id_override: Optional[str] = None) -> str:
        """
        Creates a summary of the code changes using the LIGHT_MODEL.

        Args:
            diff_content: The code diff to summarize.
            model_id_override: Optional Bedrock model ID to override the default LIGHT_MODEL_ID.
        
        Returns:
            A concise summary of the changes.

        Raises:
            ValueError: If diff_content is empty or model ID is not configured.
            RuntimeError: If model invocation fails.
        """
        if not diff_content:
            raise ValueError("Diff content cannot be empty for summarization.")

        light_model_id = model_id_override or config.LIGHT_MODEL_ID
        if not light_model_id:
            raise ValueError("LIGHT_MODEL_ID is not configured for summarization.")

        prompt = f"""Human: Please provide a concise summary of the following code changes. 
Focus on the main purpose of the changes and key modifications. 
Keep the summary to a few sentences if possible.

<diff>
{diff_content}
</diff>

Assistant: Summary of changes:"""
        
        logger.info(f"Summarizing code changes using model: {light_model_id}")
        try:
            summary = self.bedrock_handler.invoke_model(
                model_id=light_model_id,
                prompt=prompt,
                max_tokens=512, # Shorter for summary
                temperature=0.7
            )
            return summary
        except Exception as e:
            logger.error(f"Error during light model summarization (model: {light_model_id}): {e}")
            raise RuntimeError(f"Light model summarization failed: {e}") from e

    def generate_review_body(self, summary: str, refined_analysis: Optional[str] = None,
                               heavy_analysis_raw: Optional[str] = None) -> str:
        """
        Formats the summary and analysis into a presentable review body (Markdown).

        Args:
            summary: The summary of changes.
            refined_analysis: The refined analysis from the DeepSeek model (optional).
            heavy_analysis_raw: The raw analysis from the heavy model (optional, as fallback or additional info).

        Returns:
            A Markdown formatted string for the review body.
        """
        if not summary:
            logger.warning("generate_review_body called with an empty summary.")
            summary = "No summary could be generated for these changes."
        
        body = f"## AI Code Review Summary 🤖\n\n{summary}\n"

        if refined_analysis:
            body += f"\n### Detailed Analysis (Refined)\n\n{refined_analysis}\n"
        elif heavy_analysis_raw:
            body += f"\n### Detailed Analysis (Raw)\n\n{heavy_analysis_raw}\n"
        else:
            body += "\nNo detailed analysis was performed or it yielded no results.\n"
        
        body += "\n---\n*This review was auto-generated by an AI assistant.*"
        return body

    def generate_release_notes(self, diff_content: str, model_id_override: Optional[str] = None) -> str:
        """
        Generates a non-technical summary of code changes suitable for release notes,
        using the LIGHT_MODEL.

        Args:
            diff_content: The code diff to summarize for release notes.
            model_id_override: Optional Bedrock model ID to override the default LIGHT_MODEL_ID.
        
        Returns:
            A concise, non-technical summary of the changes for release notes.

        Raises:
            ValueError: If diff_content is empty or model ID is not configured.
            RuntimeError: If model invocation fails.
        """
        if not diff_content:
            raise ValueError("Diff content cannot be empty for generating release notes.")

        light_model_id = model_id_override or config.LIGHT_MODEL_ID
        if not light_model_id:
            raise ValueError("LIGHT_MODEL_ID is not configured for generating release notes.")

        prompt = f"""Human: Please act as a technical writer. Based on the following code changes, draft a brief summary for release notes.
This summary should be easy for a non-technical person to understand.
Focus on what has changed from a user's perspective or what new capabilities are introduced.
Avoid jargon and technical details.
<diff>
{diff_content}
</diff>

Assistant:Release notes summary:"""
        
        logger.info(f"Generating release notes summary using model: {light_model_id}")
        try:
            summary = self.bedrock_handler.invoke_model(
                model_id=light_model_id,
                prompt=prompt,
                max_tokens=300, # Adjusted for concise release notes
                temperature=0.7
            )
            return summary
        except Exception as e:
            logger.error(f"Error during release notes generation (model: {light_model_id}): {e}")
            raise RuntimeError(f"Release notes generation failed: {e}") from e
