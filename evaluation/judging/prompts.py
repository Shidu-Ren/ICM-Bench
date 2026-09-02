# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by the ICM-Bench authors in 2026 for public semantic-equivalence evaluation.

"""Prompts shared by the public ICM-Bench answer judges."""

SEMANTIC_EQUIVALENCE_PROMPT = """You are provided with a question, a ground truth answer, and an answer from an agent model. Your task is to determine whether the ground truth answer can be logically inferred from the agent's answer, in the context of the question.

Do not directly compare the surface forms of the agent answer and the ground truth answer. Instead, assess whether the meaning expressed by the agent answer supports or implies the ground truth answer. If the ground truth can be reasonably derived from the agent answer, return "Yes". If it cannot, return "No".

Important notes:
- Do not require exact wording or matching structure.
- Semantic inference is sufficient, as long as the agent answer entails or implies the meaning of the ground truth answer, given the question.
- Only return "Yes" or "No", with no additional explanation or formatting.

Input fields:
- question: the question asked
- ground_truth_answer: the correct answer
- agent_answer: the model's answer to be evaluated

Now evaluate the following input:

Input:
- question: {question}
- ground_truth_answer: {ground_truth_answer}
- agent_answer: {agent_answer}

Output ('Yes' or 'No'):"""
