# Manual Integration Steps

Due to formatting issues with automated edits, here are the manual steps to complete the integration of rate limiting and optimization improvements.

## 1. Fix main.py imports

Replace the import section in `src/main.py` (lines 1-11) with:

```python
import json
import logging
import os
import sys
import time
from typing import Dict

from src import config # To ensure it's loaded, though individual configs are used directly
from src.bedrock_handler import BedrockHandler
from src.github_handler import GithubHandler
from src.analysis_service import AnalysisService
from src.rate_limiter import configure_global_rate_limiter
```

## 2. Add rate limiter configuration in main.py

After line ~43 (where handlers are about to be initialized), add:

```python
        # Configure global rate limiter
        # You can adjust these values via environment variables if needed
        requests_per_minute = int(os.getenv("BEDROCK_REQUESTS_PER_MINUTE", "40"))  # Conservative default
        burst_capacity = int(os.getenv("BEDROCK_BURST_CAPACITY", "8"))
        configure_global_rate_limiter(requests_per_minute=requests_per_minute, burst_capacity=burst_capacity)
        logger.info(f"Configured rate limiter: {requests_per_minute} req/min, burst: {burst_capacity}")
```

## 3. Fix individual file analysis loop in main.py

Replace the individual file analysis section (around lines 190-210) with:

```python
        # --- Perform Individual File Analysis ---
        individual_file_analyses: Dict[str, str] = {}
        if changed_files: # Ensure changed_files is populated
            logger.info(f"Starting individual analysis for {len(changed_files)} changed files...")
            for index, file_info in enumerate(changed_files):
                filename = file_info.get("filename")
                file_patch = file_info.get("patch") # This is the diff for the individual file
                
                if filename and file_patch:
                    try:
                        logger.info(f"Performing analysis for file: {filename} ({index + 1}/{len(changed_files)})")
                        file_analysis_result = analysis_s.analyze_individual_file_diff(file_patch=file_patch, filename=filename)
                        if file_analysis_result:
                            individual_file_analyses[filename] = file_analysis_result
                            logger.info(f"Completed analysis for file: {filename}. Result length: {len(file_analysis_result)}")
                        else:
                            logger.info(f"Analysis for file: {filename} returned no content.")
                        
                        # Add delay between file analyses to spread out API calls
                        if index < len(changed_files) - 1:  # Don't delay after the last file
                            delay_seconds = 1.5  # Configurable delay between file analyses
                            logger.debug(f"Adding {delay_seconds}s delay before analyzing next file")
                            time.sleep(delay_seconds)
                        
                    except Exception as e:
                        logger.error(f"Error during analysis of file {filename}: {e}. This file's analysis will be skipped.")
                        individual_file_analyses[filename] = f"Could not analyze {filename} due to an error: {str(e)}"
                elif filename and not file_patch:
                    logger.info(f"Skipping analysis for file {filename} as it has no patch content (e.g., binary, renamed, or mode change only). ")        
        else:
            logger.info("No changed files data available to perform individual file analysis.")
```

## 4. Update AnalysisService methods

### Fix analyze_code_changes method in src/analysis_service.py

Replace the invoke_model call (around line 56) with:

```python
        logger.info(f"Analyzing code changes using model: {heavy_model_id}")
        try:
            analysis = self.bedrock_handler.invoke_model(
                model_id=heavy_model_id,
                prompt=prompt,
                analysis_type='heavy_analysis',  # Dynamic token calculation
                temperature=0.5  
            )
            return analysis
        except Exception as e:
            logger.error(f"Error during heavy model analysis (model: {heavy_model_id}): {e}")
            raise RuntimeError(f"Heavy model analysis failed: {e}") from e
```

### Update analyze_heavy_model_output method

Replace the invoke_model call (around line 105) with:

```python
            raw_structured_output = self.bedrock_handler.invoke_model(
                model_id=deepseek_model_id,
                prompt=prompt,
                analysis_type='structured_extraction',  # Dynamic token calculation
                temperature=0.3
            )
```

### Update summarize_changes method

Replace the invoke_model call (around line 245) with:

```python
            summary = self.bedrock_handler.invoke_model(
                model_id=light_model_id,
                prompt=prompt,
                analysis_type='summary',  # Dynamic token calculation
                temperature=0.7
            )
```

### Update generate_release_notes method

Replace the invoke_model call (around line 290) with:

```python
            summary = self.bedrock_handler.invoke_model(
                model_id=light_model_id,
                prompt=prompt,
                analysis_type='release_notes',  # Dynamic token calculation
                temperature=0.7
            )
```

### Update analyze_individual_file_diff method

Replace the invoke_model call (around line 365) with:

```python
            analysis = self.bedrock_handler.invoke_model(
                model_id=heavy_model_id,
                prompt=prompt,
                analysis_type='individual_file',  # Dynamic token calculation
                temperature=0.5  
            )
```

## 5. Optional Environment Variables

Add these to your `.env` file for fine-tuning:

```bash
# Rate limiting configuration
BEDROCK_REQUESTS_PER_MINUTE=40
BEDROCK_BURST_CAPACITY=8

# Token calculation overrides (optional)
HEAVY_ANALYSIS_BASE_TOKENS=1500
HEAVY_ANALYSIS_MAX_TOKENS=6000
INDIVIDUAL_FILE_BASE_TOKENS=1000
INDIVIDUAL_FILE_MAX_TOKENS=5000
```

## 6. Update GitHub Action Workflow (if needed)

If you want to expose rate limiting configuration in the GitHub Action, add these inputs to `action.yml`:

```yaml
inputs:
  bedrock_requests_per_minute:
    description: 'Maximum Bedrock API requests per minute'
    required: false
    default: '40'
  bedrock_burst_capacity:
    description: 'Bedrock API burst capacity'
    required: false
    default: '8'
```

And pass them as environment variables in the workflow.

## 7. Verify Integration

After making these changes:

1. **Test with a small PR** to ensure basic functionality works
2. **Test with a large multi-file PR** to verify rate limiting and delays
3. **Check logs** for token calculation details and rate limiter status
4. **Monitor AWS costs** to confirm token usage optimization

## Benefits After Integration

- **Dynamic token allocation** based on content complexity
- **Automatic rate limiting** prevents throttling errors
- **Staggered API calls** distribute load over time
- **Enhanced error handling** with exponential backoff
- **Comprehensive logging** for monitoring and debugging

The system will now handle both small single-file changes and large complex PRs with optimal efficiency and reliability. 