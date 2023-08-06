import folium
import openai
import requests
import streamlit as st
import tiktoken
import streamlit_folium
import requests
import os

# Set the OpenAI API key
openai.api_key = os.environ.get("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")

# Chat template string, to be used for generating Overpass API queries
CHAT_TEMPLATE = """Assistant is an expert Geo-Coder assistant.

For each question that the user supplies, the assistant will reply with:
(1) A statement consenting to help.
(2) Think of it in steps:
Step 1) Use this line of code to get the data:
``` python
req = requests.get('https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=GeoJSON')
with open('/dbfs/tmp/nyc_taxi_zones.geojson', 'wb') as f:f.write(req.content)```

Step 2) Create a dataframe using the exact code below: 
```python
neighbourhoods = (
  spark.read
    .option("multiline", "true")
    .format("json")
    .load("dbfs:/tmp/nyc_taxi_zones.geojson")

    # Extract geoJSON values for shapes
    .select("type", explode(col("features")).alias("feature"))
    .select("type", col("feature.properties").alias("properties"), to_json(col("feature.geometry")).alias("geom_json"))

    # Mosaic internal representation
    .withColumn("geom_internal", mos.st_geomfromgeojson("geom_json"))

    # WKT representation
    .withColumn("geom_wkt", mos.st_aswkt(col("geom_internal")))

    # WKB representation
    .withColumn("geom_wkb", mos.st_aswkb(col("geom_internal")))

   # Limit to only 1 shape
   .limit(1)
)
```

Step 3)Tesselate the spark dataframe using the exact code below:
```python
neighbourhood_chips = (neighbourhoods
                       .limit(1)
                       .select(mos.grid_tessellateexplode("geom_internal", lit(9)))
                       .select("index.*")
                    )
```

Step 4) We are wanting to execute code from a string that contains both python code and jupyter commands. This might difficult so we need to handle them exactly like the code below. Line is code that you want to execute:
```python
kepler_code = 
%%mosaic_kepler 
neighbourhood_chips "index_id" "h3"

lines = kepler_code.strip().split("\n")

for line in lines:
    # Check if the line starts with a magic command
    if line.startswith("%%"):
        # Execute the magic command
        get_ipython().run_cell_magic(line.split()[0].replace("%%", ""), "", "\n".join(lines[1:]))
        break
    else:
(3) A fun fact relating to the question, or a very funny joke or pun related to the question. The joke or pun should also relate to maps, geospatial tech, geography or similar concepts. There is no need to label the fact, joke, or pun.

Assistant has a whimsical personality. Assistant will reply with a geospatial themed joke or a pun if the user asks a question that is not relevant to the API.

{history}
Human: {human_input}
Assistant:"""

# Reader template string, to be used for generating text responses drawing on Overpass API responses
READER_TEMPLATE = """Read the following Overpass API response carefully. Use the information in it to answer the prompt "{prompt}" Your answer should not mention the words "API" or "Overpass." Your answer should sound like it was spoken by someone with personal knowledge of the question's answer. Your answer should be very concise, but also informative and fun. Format any names or places you get from the API response as bold text in Markdown.
Overpass API Response:
Answer: {response}
"""

#--------------- constants for direct request prompt generation  ---------------
direct_request_role = graph_role

direct_request_task_prefix = r'Write a Python program'

direct_request_reply_exmaple = """
```python',
%%mosaic_kepler
neighbourhood_chips "index_id" "h3"
```
"""
#--------------- constants for direct request prompt generation  ---------------
direct_request_role = graph_role

direct_request_task_prefix = r'Write a Python program'

direct_request_reply_exmaple = """
```python',
%%mosaic_kepler
neighbourhood_chips "index_id" "h3"
```
"""
direct_request_requirement = [
                        'DO NOT change the given variable names and paths.',
                        'Put your reply into a Python code block(enclosed by ```python and ```), NO explanation or conversation outside the code block.',
                        "Generate descriptions for input and output arguments.",
                        "Note module 'pandas' has no attribute 'StringIO'",
                        "Use the latest Python module methods.",
                        "DO NOT reproject or set spatial data(e.g., GeoPandas Dataframe) if only one layer involved.",
                        "Map projection conversion is only conducted for spatial data layers such as Mosaic Kepler. DataFrame loaded from a CSV file does not have map projection information.",
                        "If join DataFrame and GeoDataFrame, using common columns, DO NOT convert DataFrame to GeoDataFrame.",
                        "When joining tables, convert the involved columns to string type without leading zeros. ",
                        "When doing spatial joins, remove the duplicates in the results. Or please think about whether it needs to be removed.",
                        "Graphs or maps need to show the unit.",
                        "Remember the variable, column, and file names used in ancestor functions when using them, such as joining tables or calculating.",
                        # "Show a progressbar (e.g., tqdm in Python) if loop more than 200 times, also add exception handling for loops to make sure the loop can run.",
                        "When crawl the webpage context to ChatGPT, using Beautifulsoup to crawl the text only, not all the HTML file.",
                        "If using GeoPandas for spatial joining, the arguements are: geopandas.sjoin(left_df, right_df, how='inner', predicate='intersects', lsuffix='left', rsuffix='right', **kwargs), how: default ‘inner’, use intersection of keys from both dfs; retain only left_df geometry column; ‘left’: use keys from left_df, retain only left_df geometry column. ",
                        # "Drop rows with NaN cells, i.e., df.dropna(), before using Pandas or GeoPandas columns for processing (e.g. join or calculation).",
                        "The program is executable, put it in a function named 'direct_solution()' then run it, but DO NOT use 'if __name__ == '__main__:' statement because this program needs to be executed by exec().",
                        "Generate simple code"
                        ]

--------------- constants for direct program review prompt generation  ---------------
direct_review_role = graph_role

direct_review_task_prefix = r'Review the code of a program to determine whether the code meets its associated requirements. If not, correct it then return the complete corrected code. '

direct_review_requirement = [
                        'Review the code very carefully to ensure its correctness and robustness.',
                        'Elaborate your reasons for revision.',
                        'If the code has no error, and you do not need to modify the code, DO NOT return code, return "PASS" only, without any other explanation or description.',
                        'If you modified the code, return the complete corrected program. All returned code need to be inside only one Python code block (enclosed by ```python and ```)',
                        'DO NOT use more than one Python code blocks in your reply, because I need to extract the complete Python code in the Python code block.',
                        'Pay extra attention on file name, table field name, spatial analysis parameters, map projections, and NaN cells removal in the used Pandas columns.',
                        'Pay extra attention on the common field names when joining Pandas DataFrame.',
                        'The given code might has error in mapping or visualization when using GeoPandas or Matplotlib packages.',
                        #
                        ]
OVERPASS_API_URL = "https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=GeoJSON"

import LLM_Geo_Constants as constants
import helper
import os
import requests
import networkx as nx
import pandas as pd
import geopandas as gpd
# from pyvis.network import Network
import openai
import pickle
import time
import sys
import traceback


class Solution():
    """

    """
    def __init__(self, 
                 task, 
                 task_name,
                 save_dir,                
                 role=constants.graph_role,
                 model=r"gpt-3.5-turbo",
                 # model=r"gpt-3.5-turbo",
                 data_locations=[],
                 stream=True,
                 verbose=True,
                ):        
        self.task = task        
        self.solution_graph = None
        self.graph_response = None
        self.role = role
        self.data_locations=data_locations
        self.task_name = task_name   
        self.save_dir = save_dir
        self.code_for_graph = ""
        self.graph_file = os.path.join(self.save_dir, f"{self.task_name}.graphml")
        self.source_nodes = None
        self.sink_nodes = None
        self.operations = []  # each operation is an element:
        # {node_name: "", function_descption: "", function_definition:"", return_line:""
        # operation_prompt:"", operation_code:""}
        self.assembly_prompt = ""
        
        self.parent_solution = None
        self.model = model
        self.stream = stream
        self.verbose = verbose

        self.assembly_LLM_response = ""
        self.code_for_assembly = ""
        self.graph_prompt = ""
         
        self.data_locations_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(self.data_locations)])     
        
        graph_requirement = constants.graph_requirement.copy()
        graph_requirement.append(f"Save the network into GraphML format, save it at: {self.graph_file}")
        graph_requirement_str =  '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(graph_requirement)])
        
        graph_prompt = f'Your role: {self.role} \n\n' + \
               f'Your task: {constants.graph_task_prefix} \n {self.task} \n\n' + \
               f'Your reply needs to meet these requirements: \n {graph_requirement_str} \n\n' + \
               f'Your reply example: {constants.graph_reply_exmaple} \n\n' + \
               f'Data locations (each data is a node): {self.data_locations_str} \n'
        self.graph_prompt = graph_prompt

        # self.direct_request_prompt = ''
        self.direct_request_LLM_response = ''
        self.direct_request_code = ''

        self.chat_history = [{'role': 'system', 'content': role}]

    def get_LLM_reply(self,
            prompt,
            verbose=True,
            temperature=1,
            stream=True,
            retry_cnt=3,
            sleep_sec=10,
            system_role=None,
            model=None,
            ):

        openai.api_key = constants.OpenAI_key

        if system_role is None:
            system_role = self.role

        if model is None:
            model = self.model

        # Query ChatGPT with the prompt
        # if verbose:
        #     print("Geting LLM reply... \n")
        count = 0
        isSucceed = False
        self.chat_history.append({'role': 'user', 'content': prompt})
        while (not isSucceed) and (count < retry_cnt):
            try:
                count += 1
                response = openai.ChatCompletion.create(
                    model=model,
                    # messages=self.chat_history,  # Too many tokens to run.
                    messages=[
                                {"role": "system", "content": constants.operation_role},
                                {"role": "user", "content": prompt},
                              ],
                    temperature=temperature,
                    stream=stream,
                )
            except Exception as e:
                # logging.error(f"Error in get_LLM_reply(), will sleep {sleep_sec} seconds, then retry {count}/{retry_cnt}: \n", e)
                print(f"Error in get_LLM_reply(), will sleep {sleep_sec} seconds, then retry {count}/{retry_cnt}: \n",
                      e)
                time.sleep(sleep_sec)

        response_chucks = []
        if stream:
            for chunk in response:
                response_chucks.append(chunk)
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content is not None:
                    if verbose:
                        print(content, end='')
        else:
            content = response["choices"][0]['message']["content"]
            # print(content)
        print('\n\n')
        # print("Got LLM reply.")

        response = response_chucks  # good for saving

        content = helper.extract_content_from_LLM_reply(response)

        self.chat_history.append({'role': 'assistant', 'content': content})

        return response


    def get_LLM_response_for_graph(self, execuate=True):
        # self.chat_history.append()
        response = self.get_LLM_reply(
                                        prompt=self.graph_prompt,
                                        system_role=self.role,
                                        model=self.model,
                                         )
        self.graph_response = response
        try:
            self.code_for_graph = helper.extract_code(response=self.graph_response, verbose=False)
        except Exception as e:
            self.code_for_graph = ""
            print("Extract graph Python code rom LLM failed.")
        if execuate:
            exec(self.code_for_graph)
            self.load_graph_file()
        return self.graph_response
        
    def load_graph_file(self, file=""):
        G = None
        if os.path.exists(file):
            self.graph_file = file
            G = nx.read_graphml(self.graph_file)
        else:
            
            if file == "" and os.path.exists(self.graph_file):
                G = nx.read_graphml(self.graph_file)
            else:
                print("Do not find the given graph file:", file)
                return None
        
        self.solution_graph = G
        
        self.source_nodes = helper.find_source_node(self.solution_graph)
        self.sink_nodes = helper.find_sink_node(self.solution_graph)
         
        return self.solution_graph 

    @property
    def operation_node_names(self):
        opera_node_names = []
        assert self.solution_graph, "The Soluction class instance has no solution graph. Please generate the graph"
        for node_name in self.solution_graph.nodes():
            node = self.solution_graph.nodes[node_name]
            if node['node_type'] == 'operation':
                opera_node_names.append(node_name)
        return opera_node_names

    def get_ancestor_operations(self, node_name):
        ancestor_operation_names = []
        ancestor_node_names = nx.ancestors(self.solution_graph, node_name)
        # for ancestor_node_name in ancestor_node_names:
        ancestor_operation_names = [node_name for node_name in ancestor_node_names if node_name in self.operation_node_names]

        ancestor_operation_nodes = []
        for oper in self.operations:
            oper_name = oper['node_name']
            if oper_name in ancestor_operation_names:
                ancestor_operation_nodes.append(oper)

        return ancestor_operation_nodes

    def get_descendant_operations(self, node_name):
        descendant__operation_names = []
        descendant_node_names = nx.descendants(self.solution_graph, node_name)
        # for descendant_node_name in descendant_node_names:
        descendant__operation_names = [node_name for node_name in descendant_node_names if node_name in self.operation_node_names]
        # descendant_codes = '\n'.join([oper['operation_code'] for oper in descendant_node_names])
        descendant_operation_nodes = []
        for oper in self.operations:
            oper_name = oper['node_name']
            if oper_name in descendant__operation_names:
                descendant_operation_nodes.append(oper)

        return descendant_operation_nodes

    def get_descendant_operations_definition(self, descendant_operations):

        keys = ['node_name', 'description', 'function_definition', 'return_line']
        operation_def_list = []
        for node in descendant_operations:
            operation_def = {key: node[key] for key in keys}
            operation_def_list.append(str(operation_def))
        defs = '\n'.join(operation_def_list)
        return defs

    def get_prompt_for_an_opearation(self, operation):
        assert self.solution_graph, "Do not find solution graph!"
        # operation_dict = function_def.copy()

        node_name = operation['node_name']

        # get ancestors code
        ancestor_operations = self.get_ancestor_operations(node_name)
        ancestor_operation_codes = '\n'.join([oper['operation_code'] for oper in ancestor_operations])
        descendant_operations = self.get_descendant_operations(node_name)
        descendant_defs = self.get_descendant_operations_definition(descendant_operations)
        descendant_defs_str = str(descendant_defs)

        pre_requirements = [
            f'The function description is: {operation["description"]}',
            f'The function definition is: {operation["function_definition"]}',
            f'The function return line is: {operation["return_line"]}'
        ]

        operation_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
            pre_requirements + constants.operation_requirement)])

        operation_prompt = f'Your role: {constants.operation_role} \n\n' + \
                           f'operation_task: {constants.operation_task_prefix} {operation["description"]} \n\n' + \
                           f'This function is one step to solve the question/task: {self.task} \n\n' + \
                           f"This function is a operation node in a solution graph for the question/task, the Python code to build the graph is: \n{self.code_for_graph} \n\n" + \
                           f'Data locations: {self.data_locations_str} \n\n' + \
                           f'Your reply example: {constants.operation_reply_exmaple} \n\n' + \
                           f'Your reply needs to meet these requirements: \n {operation_requirement_str} \n\n' + \
                           f"The ancestor function code is (need to follow the generated file names and attribute names): \n {ancestor_operation_codes} \n\n" + \
                           f"The descendant function (if any) definitions for the question are (node_name is function name): \n {descendant_defs_str}"

        operation['operation_prompt'] = operation_prompt
        return operation_prompt
        # self.operations.append(operation_dict)
    # def get_prompts_for_operations(self):  ######## Not use ###########
    #     assert self.solution_graph, "Do not find solution graph!"
    #     def_list, data_node_list = helper.generate_function_def_list(self.solution_graph)
    #
    #
    #     for idx, function_def in enumerate(def_list):
    #         operation_dict = function_def.copy()
    #
    #         node_name = function_def['node_name']
    #
    #         # get ancestors code
    #         ancestor_operations = self.get_ancestor_operations(node_name)
    #         ancestor_operation_codes = '\n'.join([oper['operation_code'] for oper in ancestor_operations])
    #         descendant_operations = self.get_descendant_operations(node_name)
    #         descendant_defs = self.get_descendant_operations_definition(descendant_operations)
    #
    #         pre_requirements = [
    #                             f'The function description is: {function_def["description"]}',
    #                             f'The function definition is: {function_def["function_definition"]}',
    #                             f'The function return line is: {function_def["return_line"]}'
    #                            ]
    #
    #         operation_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
    #             pre_requirements + constants.operation_requirement)])
    #
    #         operation_prompt = f'Your role: {constants.operation_role} \n' + \
    #                            f'operation_task: {constants.operation_task_prefix} {function_def["description"]} \n' + \
    #                            f'This function is one step to solve the question/task: {self.task} \n' + \
    #                            f"This function is a operation node in a solution graph for the question/task, the Python code to build the graph is: \n{self.code_for_graph} \n" + \
    #                            f'Data locations: {self.data_locations_str} \n' + \
    #                            f'Reply example: {constants.operation_reply_exmaple} \n' + \
    #                            f'Your reply needs to meet these requirements: \n {operation_requirement_str} \n \n' + \
    #                            f"The ancestor function code is (need to follow the generated file names and attribute names): \n {ancestor_operation_codes}" + \
    #                            f"The descendant function definitions for the question are (node_name is function name): \n {descendant_defs}"
    #
    #
    #         operation_dict['operation_prompt'] = operation_prompt
    #         self.operations.append(operation_dict)
    #     return self.operations

    # initial the oepartion list
    def initial_operations(self):
        self.operations = []
        operation_names = self.operation_node_names
        for node_name in operation_names:
            function_def_returns = helper.generate_function_def(node_name, self.solution_graph)
            self.operations.append(function_def_returns)
    def get_LLM_responses_for_operations(self, review=True):
        # def_list, data_node_list = helper.generate_function_def_list(self.solution_graph)
        self.initial_operations()
        for idx, operation in enumerate(self.operations):
            node_name = operation['node_name']
            print(f"{idx + 1} / {len(self.operations)}, LLM is generating code for operation node: {operation['node_name']}")
            prompt = self.get_prompt_for_an_opearation(operation)

            response = self.get_LLM_reply(
                          prompt=prompt,
                          system_role=constants.operation_role,
                          model=self.model,
                          # model=r"gpt-4",
                         )
            # print(response)
            operation['response'] = response
            try:
                operation_code = helper.extract_code(response=operation['response'], verbose=False)
            except Exception as e:
                operation_code = ""
            operation['operation_code'] = operation_code

            if review:
                operation = self.ask_LLM_to_review_operation_code(operation)
            
        return self.operations


    def prompt_for_assembly_program(self):
        all_operation_code_str = '\n'.join([operation['operation_code'] for operation in self.operations])
        # operation_code = solution.operations[-1]['operation_code']
        # assembly_prompt = f"" + \

        assembly_requirement = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(constants.assembly_requirement)])

        assembly_prompt = f"Your role: {constants.assembly_role} \n\n" + \
                          f"Your task is: use the given Python functions, return a complete Python program to solve the question: \n {self.task}" + \
                          f"Requirement: \n {assembly_requirement} \n\n" + \
                          f"Data location: \n {self.data_locations_str} \n" + \
                          f"Code: \n {all_operation_code_str}"
        
        self.assembly_prompt = assembly_prompt
        return self.assembly_prompt
    
    
    def get_LLM_assembly_response(self, review=True):
        self.prompt_for_assembly_program()
        assembly_LLM_response = helper.get_LLM_reply(self.assembly_prompt,
                          system_role=constants.assembly_role,
                          model=self.model,
                          # model=r"gpt-4",
                         )
        self.assembly_LLM_response = assembly_LLM_response
        self.code_for_assembly = helper.extract_code(self.assembly_LLM_response)
        
        try:
            code_for_assembly = helper.extract_code(response=self.assembly_LLM_response, verbose=False)
        except Exception as e:
                code_for_assembly = ""
                
        self.code_for_assembly = code_for_assembly

        if review:
            self.ask_LLM_to_review_assembly_code()
        
        return self.assembly_LLM_response
    
    def save_solution(self):
#         , graph=True
        new_name = os.path.join(self.save_dir, f"{self.task_name}.pkl")
        with open(new_name, "wb") as f:
            pickle.dump(self, f)

    def get_solution_at_one_time(self):
        pass

    @property
    def direct_request_prompt(self):

        direct_request_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(
            constants.direct_request_requirement)])

        direct_request_prompt = f'Your role: {constants.direct_request_role} \n' + \
                                f'Your task: {constants.direct_request_task_prefix} to address the question or task: {self.task} \n' + \
                           f'Location for data you may need: https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=GeoJSON' +"\n" + \
                           f'Your reply needs to meet these requirements: \n {direct_request_requirement_str} \n'
        return direct_request_prompt

    def get_direct_request_LLM_response(self, review=True):

        response = helper.get_LLM_reply(prompt=self.direct_request_prompt,
                                        model=self.model,
                                        stream=self.stream,
                                        verbose=self.verbose,
                                        )

        self.direct_request_LLM_response = response

        self.direct_request_code = helper.extract_code(response=response)

        if review:
            self.ask_LLM_to_review_direct_code()

        return self.direct_request_LLM_response

    def execute_complete_program(self, code: str, try_cnt: int = 10) -> str:

        count = 0
        while count < try_cnt:
            print(f"\n\n-------------- Running code (trial # {count + 1}/{try_cnt}) --------------\n\n")
            try:
                count += 1
                compiled_code = compile(code, 'Complete program', 'exec')
                exec(compiled_code, globals())  # #pass only globals() not locals()
                #!!!!    all variables in code will become global variables! May cause huge issues!     !!!!
                print("\n\n--------------- Done ---------------\n\n")
                return code

            # except SyntaxError as err:
            #     error_class = err.__class__.__name__
            #     detail = err.args[0]
            #     line_number = err.lineno
            #
            except Exception as err:

                # cl, exc, tb = sys.exc_info()

                # print("An error occurred: ", traceback.extract_tb(tb))

                if count == try_cnt:
                    print(f"Failed to execute and debug the code within {try_cnt} times.")
                    return code

                debug_prompt = self.get_debug_prompt(exception=err, code=code)
                print("Sending error information to LLM for debugging...")
                # print("Prompt:\n", debug_prompt)
                response = helper.get_LLM_reply(prompt=debug_prompt,
                                                system_role=constants.debug_role,
                                                model=self.model,
                                                verbose=True,
                                                stream=True,
                                                retry_cnt=5,
                                                )
                code = helper.extract_code(response)

        return code


    def get_debug_prompt(self, exception, code):
        etype, exc, tb = sys.exc_info()
        exttb = traceback.extract_tb(tb)  # Do not quite understand this part.
        # https://stackoverflow.com/questions/39625465/how-do-i-retain-source-lines-in-tracebacks-when-running-dynamically-compiled-cod/39626362#39626362

        ## Fill the missing data:
        exttb2 = [(fn, lnnr, funcname,
                   (code.splitlines()[lnnr - 1] if fn == 'Complete program'
                    else line))
                  for fn, lnnr, funcname, line in exttb]

        # Print:
        error_info_str = 'Traceback (most recent call last):\n'
        for line in traceback.format_list(exttb2[1:]):
            error_info_str += line
        for line in traceback.format_exception_only(etype, exc):
            error_info_str += line

        print(f"Error_info_str: \n{error_info_str}")

        # print(f"traceback.format_exc():\n{traceback.format_exc()}")

        debug_requirement_str = '\n'.join([f"{idx + 1}. {line}" for idx, line in enumerate(constants.debug_requirement)])

        debug_prompt = f"Your role: {constants.debug_role} \n" + \
                          f"Your task: correct the code of a program according to the error information, then return the corrected and completed program. \n\n" + \
                          f"Requirement: \n {debug_requirement_str} \n\n" + \
                          f"The given code is used for this task: {self.task} \n\n" + \
                          f"The data location associated with the given code: \n {self.data_locations_str} \n\n" + \
                          f"The error information for the code is: \n{str(error_info_str)} \n\n" + \
                          f"The code is: \n{code}"

        return debug_prompt

    def ask_LLM_to_review_operation_code(self, operation):
        code = operation['operation_code']
        operation_prompt = operation['operation_prompt']
        review_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.operation_review_requirement)])
        review_prompt = f"Your role: {constants.operation_review_role} \n" + \
                          f"Your task: {constants.operation_review_task_prefix} \n\n" + \
                          f"Requirement: \n{review_requirement_str} \n\n" + \
                          f"The code is: \n----------\n{code}\n----------\n\n" + \
                          f"The requirements for the code is: \n----------\n{operation_prompt} \n----------\n"

            # {node_name: "", function_descption: "", function_definition:"", return_line:""
        # operation_prompt:"", operation_code:""}
        print("LLM is reviewing the operation code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=review_prompt,
                                        system_role=constants.operation_review_role,
                                        model=self.model,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=5,
                                        )
        new_code = helper.extract_code(response)
        reply_content = helper.extract_content_from_LLM_reply(response)
        if (reply_content == "PASS") or (new_code == ""):  # if no modification.
            print("Code review passed, no revision.\n\n")
            new_code = code
        operation['code'] = new_code

        return operation

    def ask_LLM_to_review_assembly_code(self):
        code = self.code_for_assembly
        assembly_prompt = self.assembly_prompt
        review_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.assembly_review_requirement)])
        review_prompt = f"Your role: {constants.assembly_review_role} \n" + \
                          f"Your task: {constants.assembly_review_task_prefix} \n\n" + \
                          f"Requirement: \n{review_requirement_str} \n\n" + \
                          f"The code is: \n----------\n{code} \n----------\n\n" + \
                          f"The requirements for the code is: \n----------\n{assembly_prompt} \n----------\n\n"

        print("LLM is reviewing the assembly code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=review_prompt,
                                        system_role=constants.assembly_review_role,
                                        model=self.model,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=5,
                                        )
        new_code = helper.extract_code(response)
        if (new_code == "PASS") or (new_code == ""):  # if no modification.
            print("Code review passed, no revision.\n\n")
            new_code = code

        self.code_for_assembly = new_code

    def ask_LLM_to_review_direct_code(self):
        code = self.direct_request_code
        direct_prompt = self.direct_request_prompt
        review_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.direct_review_requirement)])
        review_prompt = f"Your role: {constants.direct_review_role} \n" + \
                          f"Your task: {constants.direct_review_task_prefix} \n\n" + \
                          f"Requirement: \n{review_requirement_str} \n\n" + \
                          f"The code is: \n----------\n{code} \n----------\n\n" + \
                          f"The requirements for the code is: \n----------\n{direct_prompt} \n----------\n\n"

        print("LLM is reviewing the direct request code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=review_prompt,
                                        system_role=constants.direct_review_role,
                                        model=self.model,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=5,
                                        )
        new_code = helper.extract_code(response)
        if (new_code == "PASS") or (new_code == ""):  # if no modification.
            print("Code review passed, no revision.\n\n")
            new_code = code

        self.direct_request_code = new_code



        
    def ask_LLM_to_sample_data(self, operation_code):


        sampling_data_requirement_str = '\n'.join(
            [f"{idx + 1}. {line}" for idx, line in enumerate(constants.sampling_data_requirement)])
        sampling_data_review_prompt = f"Your role: {constants.sampling_data_role} \n" + \
                          f"Your task: {constants.sampling_task_prefix} \n\n" + \
                          f"Requirement: \n{sampling_data_requirement_str} \n\n" + \
                          f"The function code is: \n----------\n{code} \n----------\n\n" #+ \
                          # f"The requirements for the code is: \n----------\n{sampling_data_requirement_str} \n----------\n\n"

        print("LLM is reviewing the direct request code... \n")
        # print(f"review_prompt:\n{review_prompt}")
        response = helper.get_LLM_reply(prompt=sampling_data_review_prompt,
                                        system_role=constants.sampling_data_role,
                                        model=self.model,
                                        verbose=True,
                                        stream=True,
                                        retry_cnt=5,
                                        )
        code = helper.extract_code(response)
        return code
        # if (new_code == "PASS") or (new_code == ""):  # if no modification.
        #     print("Code review passed, no revision.\n\n")
        #     new_code = code

        # self.direct_request_code = new_code



# Define a function to query the Overpass API and return the JSON response
def query_overpass(query):
    payload = {"data": query}
    response = requests.post(OVERPASS_API_URL, data=payload)
    return response.json()

# Define the Streamlit app
def main():
    # Set the app title and description
    st.set_page_config(layout="wide", page_title="AskAT&TGeo", page_icon=":earth_africa:")
    st.title("AskAT&TGeo:earth_africa:")
    st.write("Hello! I'm a Geospatial AI assistant. For any question you ask in the textbox below, "
             "I'll generate an Hexagon query to answer your question, and plot the results on a map. "
             "I'll remember our conversation, so feel free to ask follow ups. :smile:")

    # Define the layout of the app
    col1, col2 = st.columns([1, 1])

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ""

    if 'overpass_query' not in st.session_state:
        st.session_state.overpass_query = None

    if 'prompt_history' not in st.session_state:
        st.session_state.prompt_history = ""

    # Define the query input box in the left pane
    with col1:
        chat = st.text_area("What can I help you find? :thinking_face:")

        if st.button("Ask"):
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=CHAT_TEMPLATE.format(history=st.session_state.chat_history, human_input=chat),
                temperature=0,
                max_tokens=516,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Display the response as pure text
            st.write(direct_request_LLM_response = solution.get_direct_request_LLM_response(review=True))

            # Update the history string
            st.session_state.chat_history = st.session_state.chat_history + f"Human: {chat}\nAssistant: {response['choices'][0]['text']}\n"

            # Update the prompt history string
            st.session_state.prompt_history = st.session_state.prompt_history + f"{chat} "

            # Update the Overpass query. The query is enclosed by three backticks, denoting that is a code block.
            # does the response contain a query? If so, update the query
            if "```" in response["choices"][0]["text"]:
                st.session_state.overpass_query = response["choices"][0]["text"].split("```")[1]
            else:
                st.session_state.overpass_query = None

            # Define the query button in the left pane
            with col2:

                if st.session_state.overpass_query:
                    # Query the Overpass API
                    response = query_overpass(st.session_state.overpass_query)

                    # Check if the response is valid
                    if "elements" in response and len(response["elements"]) > 0:
                        # Create a new Folium map in the right pane
                        m = folium.Map(location=[response["elements"][0]["lat"], response["elements"][0]["lon"]], zoom_start=11)

                        # Add markers for each element in the response
                        for element in response["elements"]:
                            if "lat" in element and "lon" in element:
                                folium.Marker([element["lat"], element["lon"]]).add_to(m)

                        # Display the map
                        streamlit_folium.folium_static(m)

                        # If the request for summary of the API response is shorter than 1500 tokens,
                        # use the Reader model to generate a response

                        query_reader_prompt  = READER_TEMPLATE.format(prompt=st.session_state.prompt_history,
                                                                      response=str(response))
                        query_reader_prompt_tokens = len(ENC.encode(query_reader_prompt))
                        if query_reader_prompt_tokens < 1500:

                            response = openai.Completion.create(
                                model="gpt-3.5-turbo",
                                prompt=query_reader_prompt,
                                temperature=0.5,
                                max_tokens=2047 - query_reader_prompt_tokens,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0
                            )

                            # Display the response as pure text
                            st.write(response["choices"][0]["text"])
                        else:
                            st.write("The API response is too long for me to read. Try asking for something slightly more specific! :smile:")
                    else:
                        st.write("No results found :cry:")

if __name__ == "__main__":
    main()
