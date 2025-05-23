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
                analysis_type='heavy_analysis',  # Dynamic token calculation               
                  temperature=0.5              )            
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

        prompt = f"""Human: Extract actionable code review suggestions from the analysis below. You MUST respond with ONLY a valid JSON array - no explanations, no conversation, no markdown blocks.

Original Code Diff:
<diff>
{diff_content}
</diff>

Analysis:
<previous_analysis>
{heavy_model_output}
</previous_analysis>

Required JSON format (respond with this ONLY):
[
  {{"file_path": "string", "line": integer, "suggestion": "string"}}
]

Rules:
- ONLY output valid JSON array
- Each suggestion must have: file_path (string), line (integer), suggestion (string)
- Use actual file paths and line numbers from the diff
- If no actionable suggestions found, return: []
- NO explanations, NO conversation, NO markdown

Assistant:JSON Output:"""

        logger.info(f"Extracting structured suggestions using model: {deepseek_model_id}")
        try:
            raw_structured_output = self.bedrock_handler.invoke_model(
                model_id=deepseek_model_id,
                prompt=prompt,
                analysis_type='structured_extraction',  # Dynamic token calculation
                temperature=0.1  # Lower temperature for more deterministic JSON output
            )
            
            logger.debug(f"Raw structured output from model {deepseek_model_id}: {raw_structured_output}")
            
            # Attempt to extract JSON, handling potential markdown code blocks
            json_str_to_parse = raw_structured_output.strip()

            # Remove markdown code blocks if present
            if json_str_to_parse.startswith("```json"):
                json_str_to_parse = json_str_to_parse[len("```json"):].strip()
                if json_str_to_parse.endswith("```"):
                    json_str_to_parse = json_str_to_parse[:-len("```")].strip()
            elif json_str_to_parse.startswith("```"):
                json_str_to_parse = json_str_to_parse[len("```"):].strip()
                if json_str_to_parse.endswith("```"):
                    json_str_to_parse = json_str_to_parse[:-len("```")].strip()

            # If the response doesn't start with '[', try to find JSON array within the text
            if not json_str_to_parse.startswith('['):
                logger.warning(f"Response doesn't start with '[', attempting to extract JSON from: {json_str_to_parse[:100]}...")
                json_start_index = json_str_to_parse.find('[')
                if json_start_index != -1:
                    json_end_index = json_str_to_parse.rfind(']')
                    if json_end_index != -1 and json_start_index < json_end_index:
                        json_str_to_parse = json_str_to_parse[json_start_index:json_end_index+1]
                        logger.debug(f"Extracted JSON substring: {json_str_to_parse[:200]}...")
                    else:
                        logger.error(f"Could not find valid JSON array in model output: {json_str_to_parse[:200]}...")
                        return []
                else:
                    logger.error(f"No JSON array found in model output: {json_str_to_parse[:200]}...")
                    return []

            try:
                # Try to parse the cleaned output
                parsed_suggestions = json.loads(json_str_to_parse)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}. Content: {json_str_to_parse[:200]}...")
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
                analysis_type='summary',  # Dynamic token calculation
                temperature=0.7
            )
            return summary
        except Exception as e:
            logger.error(f"Error during light model summarization (model: {light_model_id}): {e}")
            raise RuntimeError(f"Release notes generation failed: {e}") from e

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

        # Sanitize potentially error-containing analysis strings before including them
        sanitized_refined_analysis = config.sanitize_model_arn_in_message(refined_analysis) if refined_analysis else None
        sanitized_heavy_analysis_raw = config.sanitize_model_arn_in_message(heavy_analysis_raw) if heavy_analysis_raw else None

        if sanitized_refined_analysis:
            body += f"\n### Detailed Analysis (Refined)\n\n{sanitized_refined_analysis}\n"
        elif sanitized_heavy_analysis_raw:
            body += f"\n### Detailed Analysis (Raw)\n\n{sanitized_heavy_analysis_raw}\n"
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
                analysis_type='release_notes',  # Dynamic token calculation
                temperature=0.7
            )
            return summary
        except Exception as e:
            logger.error(f"Error during release notes generation (model: {light_model_id}): {e}")
            raise RuntimeError(f"Release notes generation failed: {e}") from e

    def analyze_individual_file_diff(self, file_patch: str, filename: str, model_id_override: Optional[str] = None) -> str:
        """
        Analyzes a single file's diff content using the HEAVY_MODEL.

        Args:
            file_patch: The code diff/patch for the specific file.
            filename: The name of the file being analyzed.
            model_id_override: Optional Bedrock model ID to override the default HEAVY_MODEL_ID.

        Returns:
            The analysis result for the individual file.
        
        Raises:
            ValueError: If file_patch or filename is empty, or model ID is not configured.
            RuntimeError: If model invocation fails.
        """
        if not file_patch:
            # This can happen for empty files or files with only mode changes etc.
            logger.info(f"File patch for '{filename}' is empty. Skipping analysis for this file.")
            return ""
        if not filename:
            raise ValueError("Filename cannot be empty for individual file analysis.")
        
        heavy_model_id = model_id_override or config.HEAVY_MODEL_ID
        if not heavy_model_id:
            raise ValueError(f"HEAVY_MODEL_ID is not configured for individual file analysis of '{filename}'.")

        prompt = f"""Human: You are an expert code reviewer. Please analyze ONLY the following code changes for the file `{filename}`. 
Focus on potential bugs, style issues, security vulnerabilities, and areas for improvement specific to these changes in this file. 
Do not make assumptions about code outside this specific patch.

File: `{filename}`
<patch>
{file_patch}
</patch>

Assistant:Analysis for file `{filename}`:"""
        
        logger.info(f"Analyzing individual file changes for: {filename} using model: {heavy_model_id}")
        try:
            analysis = self.bedrock_handler.invoke_model(
                model_id=heavy_model_id,
                prompt=prompt,
                analysis_type='individual_file',  # Dynamic token calculation
                temperature=0.5  
            )
            return analysis
        except Exception as e:
            logger.error(f"Error during individual file analysis for '{filename}' (model: {heavy_model_id}): {e}")
            # Don't let a single file analysis failure stop the whole PR review. Log and return empty.
            # Or, re-raise if individual file failures should be critical.
            # For now, returning empty string to allow process to continue for other files/overall review.
            # Sanitize the error message before returning it
            return config.sanitize_model_arn_in_message(f"Error analyzing file {filename}: {str(e)}") # Return sanitized error message
