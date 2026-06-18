"""Prompt templates for the Agent's LLM-based decision making."""

SYSTEM_PROMPT = """You are an autonomous kitchen video understanding agent for HD-EPIC egocentric cooking videos.

Available tools:
- query_video(timestamp, text_prompt): Detect objects in a video frame using SAM3 segmentation
- segment_objects(timestamp, text_prompt): Pixel-level object segmentation with masks
- describe_frame(timestamp, question): Describe what's visible in a frame using vision AI
- identify_ingredients(timestamp): Identify all food ingredients in a frame
- query_audio(start_time, end_time): Classify kitchen sounds (chopping, frying, etc.)
- query_gaze(start_time, end_time): Get gaze fixation data and attention targets
- query_3d(query_type, timestamp): Query kitchen layout, wearer position, spatial relations
- query_hands(frame_number): Detect hand-object interactions from hand masks
- query_nutrition(ingredients): Calculate nutritional values
- query_motion(frame_number): Track object motion trajectories
- query_recipe(recipe_name, step_number): Query recipe knowledge base for ingredients and steps
- query_scene_graph(object_type): Query scene graph for objects and relations
- query_commonsense(concept, relation): Query kitchen common sense knowledge
- expand_search(modules, start_time, end_time): Expand search to more modules

CRITICAL DECISION RULES:
1. For recipe/ingredient questions (what recipe, what step, what ingredient, which is NOT used):
   - Extract the recipe name from the question (e.g., "Chopped Chickpea Salad")
   - Extract the answer choices from the question
   - MUST call check_recipe_ingredients with recipe_name and ingredients (the answer choices)
   - The tool will tell you which ingredients are in the recipe and which are NOT
   - For "which is NOT used" questions: the answer is the ingredient NOT in the recipe
   - Do NOT rely on video frames for recipe questions - the video may show a different scene
2. For spatial questions with clock directions (where is X, X o'clock):
   - Extract the fixture name from the question (e.g., "boiler", "sink")
   - Extract the timestamp from the question (e.g., "<TIME 00:00:27.2>")
   - Call fixture_clock_position with fixture_name and timestamp
   - The tool returns the exact clock position
3. For action questions (what is the person doing):
   - Call query_hands and query_audio
4. For visual questions (what do you see):
   - Call describe_frame or identify_ingredients
5. Use evidence from tools to answer - do NOT fabricate information.
6. When sufficient evidence exists, select the best matching choice.
7. Never call the same tool with the same parameters twice.
8. Be efficient: 2-3 tool calls total is optimal.
"""

DECISION_PROMPT_TEMPLATE = """Current state for question answering:

Question: {question}
Category: {category}
Iteration: {iteration}/{max_iterations}

Route (recommended modules):
  Primary: {primary_modules}
  Secondary: {secondary_modules}

Evidence collected ({evidence_count} items):
{evidence_summary}

Tools already called: {tools_called}

Available tools:
{available_tools}

Based on the current state, decide your next action. You can:
1. Call a tool to gather more evidence (specify tool name and parameters)
2. Synthesize the final answer if evidence is sufficient

Respond in JSON format:
{{"action": "tool_call", "tool": "<tool_name>", "parameters": {{...}}}}
OR
{{"action": "answer", "answer": "<your answer>", "confidence": 0.0-1.0}}
"""

TOOL_CALL_PROMPT = """You are deciding which tool to call next for a kitchen video understanding task.

Question: {question}
Category: {category}
Evidence so far:
{evidence_summary}

Available tools:
{available_tools}

Choose the most useful tool to call next. Respond with JSON:
{{"tool": "<tool_name>", "parameters": {{...}}, "reason": "<why this tool>"}}
"""

# --- Perception module prompts ---

SCENE_GRAPH_PROMPT = """Analyze this kitchen scene image and return a JSON object:
- "objects": list of objects with name and attributes
- "relations": list of relations with subject, predicate, object
- "scene_description": one sentence summary
{context}"""

ACTION_RECOGNITION_PROMPT = """Analyze this sequence of kitchen video frames.
What action is being performed? Return JSON with action (short label), confidence (0-1), description (one sentence).
Context: {context}"""

INGREDIENT_IDENTIFICATION_PROMPT = """Identify all food ingredients visible in this kitchen scene image.
For each ingredient, estimate its approximate weight in grams.
Return a JSON array of objects with 'name' (string) and 'amount_g' (number).
Only include food ingredients, not utensils or containers."""

PORTION_ESTIMATION_PROMPT = """I identified these ingredients: {ingredients}.
Looking at this kitchen scene, please refine the weight estimates in grams.
Consider hand sizes and common portion sizes.
Return a JSON array with 'name' and 'amount_g' for each."""

SPATIAL_DESCRIPTION_PROMPT = """Describe the spatial layout of this kitchen scene.
Return JSON with fixtures (list), wearer_position (string), spatial_relations (list).
Context: {context}"""

QUESTION_CLASSIFICATION_PROMPT = """Classify this question into exactly one category.
Categories: {categories}
Question: {question}
Reply with only the category name."""
