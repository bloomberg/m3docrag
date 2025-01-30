# Copyright 2024 Bloomberg Finance L.P.
#
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
#
# SPDX-License-Identifier: Apache-2.0

import string

binary_page_retrieval_template = """
question: $question
Does this page have the answer to the question?
Answer only with yes or no.
""".strip()

concat_page_retrieval_template = """
question: $question
Which page is most relevant to the question?
""".strip()

concat_page_retrieval_with_answer_template = """
Find the page most relevant to the question and answer: "$question"
""".strip()

concate_page_answer_template = """
Find the answer to the question: "$question"
""".strip()

short_answer_template = """
question: $question
output only answer.
""".strip()

long_answer_template = """
question: $question
Answer the question with detailed explanation.
""".strip()


text_rag_template = """
DOCUMENTS:
$documents

QUESTION:
$question

INSTRUCTIONS:
Answer the QUESTION using the DOCUMENTS text above. Simply output the answer only.

Answer:
"""



binary_page_retrieval_template = string.Template(binary_page_retrieval_template)
concat_page_retrieval_template = string.Template(concat_page_retrieval_template)
concat_page_retrieval_with_answer_template = string.Template(concat_page_retrieval_with_answer_template)
concate_page_answer_template = string.Template(concate_page_answer_template)
short_answer_template = string.Template(short_answer_template)
long_answer_template = string.Template(long_answer_template)
text_rag_template = string.Template(text_rag_template)