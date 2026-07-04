import copy
import random
import shlex
import re
import json
import uuid

from copy import deepcopy

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers, errors, sd_models
from modules.processing import Processed, process_images
from modules.shared import state
from modules.images import image_grid, save_image
from modules.shared import opts

from parsimonious import Grammar, NodeVisitor, ParseError


def process_model_tag(tag):
    info = sd_models.get_closet_checkpoint_match(tag)
    assert info is not None, f'Unknown checkpoint: {tag}'
    return info.name


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": process_model_tag,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "rotate": process_boolean_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    rest = []

    while pos < len(args):
        arg = args[pos]

        # skip non-option arguments
        if not arg.startswith("--"):
            rest.append(arg)
            pos += 1
            continue

        # assert arg.startswith("--"), f'must start with "--": {arg}'

        # ensure if we found an arg, it has a value
        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue

        if tag == "noroot":
            res["noroot"] = True
            pos += 1
            continue

        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        val = args[pos+1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)

        pos += 2

    return res, " ".join(rest)


# ===========================================================================
# === helpers for parsing, traversing the tree
# ===========================================================================

var_grammar = Grammar(
    r"""
    full_text = ( var / var_text )*
    var = ( var_set / var_op / var_ref  )
    var_set = "{" var_name "=" full_text "}"
    var_op = "{" var_name ":" full_text "}"
    var_ref = "{" var_name "}"
    var_name = ~"[a-zA-Z_][a-zA-Z0-9_]*"
    var_text = ~"[^{}]*"
    """
)

def replace_vars(text, vars_dict=None, force_retrieve=False):
    """
    Look for variable references and operations of the following forms:
    - {var:text}: sets var to text and inserts it into the parent string
    - {var}: inserts the curent value of var into the parent string
    - {var=text}: sets var to text, but doesn't insert anything at its location

    Nested variables are supported, e.g.:
    - {color=green} {clothing: a {color} sun dress} =>
        text: "a green sun dress"
        vars dict: {"color": "green", "clothing": "a green sun dress"}
    Inner variables are replaced with values first.
    """
    if vars_dict is None:
        vars_dict = {}

    tree = var_grammar.parse(text)

    class VarVisitor(NodeVisitor):
        def visit_var_set(self, node, visited_children):
            # of the form {var_name=text}
            # this "hard" sets the variable, i.e. if it was previously defined it's overwritten
            # nothing is inserted into the parent string
            var_name = node.children[1].text
            value = "".join(str(x) for x in visited_children)
            vars_dict[var_name] = value
            return ""

        def visit_var_op(self, node, visited_children):
            # of the form {var_name:text}
            # this "soft" sets the variable, i.e. if it was previously defined it's *not* overwritten
            # the value is inserted into the parent string
            var_name = node.children[1].text
            candidate_value = "".join(str(x) for x in visited_children)
            # always retrieve from vars dict first, then resort to setting to the given value
            value = vars_dict.get(var_name, candidate_value)
            print(f"Setting variable '{var_name}' to '{value}'; {visited_children}")
            if force_retrieve or var_name not in vars_dict:
                vars_dict[var_name] = value
            return value

        def visit_var_ref(self, node, visited_children):
            # of the form {var_name}
            # simply looks up the value of the variable if set and inserts it
            # if it's not set, returns an empty string
            var_name = node.children[1].text
            return vars_dict.get(var_name, "")

        def visit_var_text(self, node, visited_children):
            # a basic text value, used as-is
            return node.text

        def generic_visit(self, node, visited_children):
            return "".join(visited_children)

    visitor = VarVisitor()

    try:
        result = visitor.visit(tree)
    except ParseError as e:
        raise ValueError(f"Error parsing variables in text: {text}") from e

    if isinstance(result, str):
        result = result.strip()

    return result, vars_dict

def text_to_tree(text, nest_delim="#", cancel_delim="x", verbose=True):
    # record state for building the tree
    last_level = 0
    base = {"text": "", "children": [], "enabled": True}
    path = [base]

    for line in text.strip().splitlines():
        # ignore empty lines
        if not line.strip():
            continue

        # split the line into prefixes and text
        matcher = r"([" + cancel_delim + "])?([" + nest_delim + r"]* )?(.*)"
        m = re.match(matcher, line.strip())
        canceller, pre, text = m.groups()

        if verbose:
            print(f"Matcher: {matcher}")
            print(f"Processing line: '{line}'")
            print(f"  Prefix: '{pre}', Text: '{text}'")
            print(f"  Last level: {last_level}")

        # determine the depth of indentation
        level = len(pre) - 1 if pre is not None else 0

        # the new node for this line; we need to figure out if it's:
        # - at the same level (last_level == level): append to curnode children
        # - below (last_level < level): append to curnode children, make new curnode
        # - above (last_level > level): make curnode-1 the new curnode, append to children
        candidate = {
            "text": text, "children": [],
            "enabled": pre is None or not canceller # not pre.startswith(cancel_delim)
        }

        if level == last_level:
            path[-1]["children"].append(candidate)
        elif level > last_level:
            # make what was the end of the children the new parent
            new_parent = path[-1]["children"][-1]
            path.append(new_parent)
            # add to new parent
            new_parent["children"].append(candidate)
        elif level < last_level:
            # pop as many levels as we've reduced
            for _ in range(last_level - level):
                path.pop()
            path[-1]["children"].append(candidate)

        # record this for the next line
        last_level = level

    return base

# do a depth-first traversal to construct lines
def collect_lines(cur, path=[], results=None, vars_dict=None, join_str=" , ", every_line_generates=False):
    """
    Collect lines from the tree, replacing variables in the text.
    The path is used to build the full text for each line.
    The results are accumulated in the results list.
    The vars_dict is used to store variable values for replacement.

    Result is a list of tuples (line, vars_dict) where:
    - line is the full text for the line, with variables replaced
    - vars_dict is the dictionary of variables for this line, with any new variables set
    """

    if not isinstance(cur, dict):
        raise ValueError("Current node must be a dictionary representing a tree node.")

    if vars_dict is None:
        vars_dict = {}

    if results is None:
        results = []

    new_vars_dict = None

    print(f"Collecting from: {cur['text']} (children: {len(cur.get('children', []))})")

    if cur["enabled"] and "children" in cur and len(cur["children"]) > 0:
        if every_line_generates:
            # force all non-leaf nodes to be treated as leaves
            full_text, new_vars_dict = replace_vars(cur['text'], vars_dict=deepcopy(vars_dict))
            sofar = [x.strip() for x in path + [full_text] if x.strip() != '']
            # ensure only non-empty lines are added
            if join_str.join(sofar).strip() != "":
                results.append((join_str.join(sofar), new_vars_dict))
                print(f"Added non-leaf node as a prompt due to every_line_generates=True: {join_str.join(sofar)}")
        elif (m := re.match(r"^[ ]*!(.*)", cur['text'])):
            # if the line starts with a '!', add it as a prompt, even if it's not a leaf
            # remove the '!' from the front
            newtext = m.group(1).strip()
            cur['text'] = newtext # and remove the '!' from the text
            print(f"Found a prompt with a shebang: {newtext}")
            # do full var replacement, since this is a leaf
            full_text, new_vars_dict = replace_vars(newtext, vars_dict=deepcopy(vars_dict))
            sofar = [x.strip() for x in path + [newtext] if x.strip() != '']
            results.append((join_str.join(sofar), new_vars_dict))

        # parse variables in the string into the vars dict, but don't retain var resolution in the text
        # (basically, this causes variable sets and refs to remain in the text)
        _, new_vars_dict = replace_vars(cur['text'], vars_dict=deepcopy(vars_dict))

        if "|" in cur["text"]:
            # split on pipes, replicate each assignment over the children
            parts = [x.strip() for x in cur["text"].split("|") if x.strip() != '']
            for part in parts:
                # parse variables in the string into the vars dict, but don't retain var resolution in the text
                _, new_vars_dict = replace_vars(part, vars_dict=deepcopy(vars_dict))
                # broadcast the part across each child
                for child in cur["children"]:
                    # descend into the children
                    collect_lines(child, path + [part], results, vars_dict=new_vars_dict, every_line_generates=every_line_generates)
        else:
            for child in cur["children"]:
                # descend into the children
                collect_lines(child, path + [cur['text']], results, vars_dict=new_vars_dict, every_line_generates=every_line_generates)
    else:
        if cur["enabled"]:
            # do full var replacement, since this is a leaf
            full_text, new_vars_dict = replace_vars(cur['text'], vars_dict=deepcopy(vars_dict))

            # it's a leaf, add it to the results
            sofar = [x.strip() for x in path + [full_text] if x.strip() != '']
            results.append((join_str.join(sofar), new_vars_dict))

    return results

# ===========================================================================
# === main script implementation
# ===========================================================================

# overrides gradio 4.40.0's improper rendering of backtick sections in markdown
# fixed in https://github.com/gradio-app/gradio/pull/1615
css = """
.gradio-container .prose pre,
.gradio-container .prose pre code,
.gradio-container .prose code {
    color: #f8f8f2 !important;
}

.gradio-container .prose pre {
    background-color: #1e1e1e !important;
    padding: 10px !important;
    border-radius: 6px !important;
}

.gradio-container .prose pre code {
    background: transparent !important;
}

.gradio-container .prose code {
    background-color: #2a2a2a !important;
    border-radius: 6px !important;
}
"""

docs = """
### Basic Usage

Lines without a prefix are concatenated with the base prompt.
                
Lines with a # prefix are concatenated with lines above them with a "," to separate them.
                
Multiple # prefixes define nesting, and only the leaves are rendered as images.
                
Each line is referred to as a "node" in tree created by the # prefixes.
A node with no children is referred to as a "leaf node".

For example, for the base prompt "a woman in a red dress":
```
sitting on a chair
# looking out the window
## moon in window
## sun in window
# reading a book
```

Will produce the following three images:
- a woman in a red dress, sitting on a chair, looking out the window, moon in window
- a woman in a red dress, sitting on a chair, looking out the window, sun in window
- a woman in a red dress, sitting on a chair, reading a book

### Variables

Lines that contain {var_name: text} will set the variable var_name to 'text'
and will insert it into the prompt at that position. Future references to {var_name} will insert the value of that variable at that position.
Using {var_name=text} will set the variable but not insert it into the prompt at that position.
These variables and assignments are available in all child nodes of the node where they are defined.

This is useful for, e.g., removing references to a character's hair color
if you want to render just their body, e.g. "1girl, {hair:short hair}, standing, posing."
could be overridden in a child node as "{hair=} closeup of torso" to remove
the reference to their hair and thus force the model to include their head.

### Broadcasting

A line with a pipe (|) in it will be split on the pipe into separate parts.
Each part will be treated as a separate node, and the tree of children of that node will
be replicated for each part. Variable assignments work as expected, too.

For example, for the base prompt "a woman in a red dress":
```
sitting on a {color:brown} chair | standing on a {color:blue} chair
# looking out the {color} window
```

Will produce the following two images:
- a woman in a red dress, sitting on a chair, looking out the brown window
- a woman in a red dress, standing on a chair, looking out the blue window

### More Controls, Flags

Other prefixes:
- ! will force the node to render even if it's a non-leaf node
- ? will omit the base prompt, useful for a "scene" image that doesn't involve the character in the base prompt

A subset of other flags:       
- `--width <x>`: set the image's width to 'x'
- `--height <x>`: set the image's height to 'x'
- `--rotate true`: swap the width and height for this specific image
- `--restore_faces <true/false>`: enables the face restoring GAN if true, disables it if false

"""

class Script(scripts.Script):
    # refs to parts of the UI; will be filled by after_component
    txt2img_prompt = None
    txt2img_neg_prompt = None
    img2img_prompt = None
    img2img_neg_prompt = None

    def after_component(self, component, **kwargs):
        elem_id = kwargs.get("elem_id", None)

        if elem_id == "txt2img_prompt":
            Script.txt2img_prompt = component
        elif elem_id == "txt2img_neg_prompt":
            Script.txt2img_neg_prompt = component
        elif elem_id == "img2img_prompt":
            Script.img2img_prompt = component
        elif elem_id == "img2img_neg_prompt":
            Script.img2img_neg_prompt = component

    def title(self):
        return "Scenario Tree"

    def _load_scenarios(self):
        print("* Loading scenarios from JSON...")

        try:
            with open("scenarios.json", "r") as fp:
                return json.load(fp)
        except:
            return {}

    def _save_scenarios(self, scenarios):
        try:
            if scenarios is None:
                raise Exception("No scenarios to save")
            with open("scenarios.json", "w") as fp:
                json.dump(scenarios, fp, indent=4)
        except:
            # make the scenarios.json and save an empty dict
            with open("scenarios.json", "w") as fp:
                json.dump({}, fp)


    def ui(self, is_img2img):
        gr.HTML(f"<style>{css}</style>")

        self.scenarios = self._load_scenarios()

        # Load last scenario text if it exists
        last_scenario = {}
        last_scenario_text = ""
        try:
            with open("_last_scenario.json", "r") as f:
                payload = f.read()
                try: 
                    last_scenario = json.loads(payload)
                    last_scenario_text = last_scenario.get("text", "")
                    last_scenario_prompt = last_scenario.get("prompt", "")
                    last_scenario_prompt_neg = last_scenario.get("negative_prompt", "")
                except json.JSONDecodeError as ex:
                    # assume it is plain text
                    print(f"Encountered error loading last scenario as JSON: {ex}")
                    last_scenario_text = payload
        except FileNotFoundError:
            pass

        checkbox_iterate = gr.Checkbox(label="Iterate seed every line", value=False, elem_id=self.elem_id("checkbox_iterate"))
        checkbox_iterate_batch = gr.Checkbox(label="Use same random seed for all lines", value=True, elem_id=self.elem_id("checkbox_iterate_batch"))
        every_line_generates = gr.Checkbox(label="Every line generates an image", value=False, elem_id=self.elem_id("every_line_generates"))
        prompt_position = gr.Radio(["start", "end"], label="Insert prompts at the", elem_id=self.elem_id("prompt_position"), value="end")
        make_combined = gr.Checkbox(label="Make a combined image containing all outputs (if more than one)", value=False)

        prompt_txt = gr.Textbox(
            label="List of prompt inputs",
            lines=3, elem_id=self.elem_id("prompt_txt"),
            value=last_scenario_text
        )

        # add a collapsible section entitled "help" containing a static text element
        help_acc = gr.Accordion("Syntax Help", open=False, elem_id=("help_acc"))
        with help_acc:
            _ = gr.Markdown(docs)

        # add read-only textboxes that show the base prompts
        # add padding under the accordion to separate it from the save/load section
        base_prompts_acc = gr.Accordion("Base prompts (saved)", open=False, elem_id=self.elem_id("base_prompts_acc"))
        with base_prompts_acc:
            base_prompt_txt = gr.Textbox(
                label="Base prompt (when saved)",
                lines=2, elem_id=self.elem_id("base_prompt_txt"),
                interactive=False,
                value=last_scenario_prompt
            )
            base_neg_prompt_txt = gr.Textbox(
                label="Base negative prompt (when saved)",
                lines=2, elem_id=self.elem_id("base_neg_prompt_txt"),
                interactive=False,
                value=last_scenario_prompt_neg
            )
            copy_base_btn = gr.Button(value="Copy base prompts to main prompt boxes", elem_id=self.elem_id("copy_base_btn"))
            _ = gr.Markdown(" ", elem_id=self.elem_id("base_prompts_padding"))

        # -----------------------
        # acquire references to base prompts, depending on mode
        # -----------------------
        if is_img2img:
            base_prompt_comp = Script.img2img_prompt
            base_neg_prompt_comp = Script.img2img_neg_prompt
        else:
            base_prompt_comp = Script.txt2img_prompt
            base_neg_prompt_comp = Script.txt2img_neg_prompt

        # -----------------------
        # saving, loading scenarios
        # -----------------------

        gr.Markdown("### Save and load scenarios", elem_id=self.elem_id("save_load_markdown"))

        # dropdown for saved scenarios
        with gr.Row():
            scenarios_dropdown = gr.Dropdown(choices=list(self.scenarios.keys()), label="Saved scenarios", show_label=False, container=False)
            load_scenario_btn = gr.Button(value="Load", scale=0)
            refresh_scenario_btn = gr.Button(value="Refresh", scale=0)

        # name, textbox for new scenarios
        with gr.Row():
            new_scenario_box = gr.Textbox(label="", show_label=False, scale=2, container=False, value="")
            scenario_save_btn = gr.Button(value="Save", scale=0)
            scenario_del_btn = gr.Button(value="Delete", scale=0)

        # add a button that loads _last_scenario.json into the prompt box
        def load_last_scenario():
            try:
                with open("_last_scenario.json", "r") as f:
                    data = f.read()
                    try:
                        payload = json.loads(data)
                        bp = payload.get("prompt", "")
                        bnp = payload.get("negative_prompt", "")
                        should_open = bool((bp or "").strip() or (bnp or "").strip())
                        return (
                            payload.get("text", ""),
                            bp,
                            bnp,
                            gr.Accordion(open=should_open),
                        )
                    except json.JSONDecodeError:
                        return (data, "", "", gr.Accordion(open=False))
            except FileNotFoundError:
                return ("", "", "", gr.Accordion(open=False))
        
        load_last_btn = gr.Button(value="Load last scenario", elem_id=self.elem_id("load_last_btn"))
        load_last_btn.click(load_last_scenario, outputs=[prompt_txt, base_prompt_txt, base_neg_prompt_txt, base_prompts_acc, ])

        # add a button that copies base_prompt_txt and base_neg_prompt_txt to the actual base prompt components
        def copy_base_prompts_to_components(base_prompt, base_neg_prompt):
            return base_prompt, base_neg_prompt

        # handlers
        def refresh_scenarios():
            self.scenarios = self._load_scenarios()
            return gr.Dropdown(choices=list(self.scenarios.keys()), label="Saved scenarios", show_label=False, container=False)

        refresh_scenario_btn.click(refresh_scenarios, outputs=[scenarios_dropdown])

        def save_scenario(new_scenario_box, prompt_txt, base_prompt, base_neg_prompt):
            if not new_scenario_box or new_scenario_box.strip() == "":
                return
            
            self.scenarios[new_scenario_box] = {
                "text": prompt_txt,
                "base_prompt": base_prompt or "",
                "base_neg_prompt": base_neg_prompt or ""
            }

            self._save_scenarios(self.scenarios)
            return gr.Dropdown(choices=list(self.scenarios.keys()), label="Saved scenarios", show_label=False, container=False)

        def delete_scenario(new_scenario_box):
            if new_scenario_box in self.scenarios:
                del self.scenarios[new_scenario_box]
                self._save_scenarios(self.scenarios)
            return gr.Dropdown(choices=list(self.scenarios.keys()), label="Saved scenarios", show_label=False, container=False)

        scenario_save_btn.click(
            save_scenario, inputs=[
                new_scenario_box, prompt_txt, base_prompt_comp, base_neg_prompt_comp
            ], outputs=[
                scenarios_dropdown
            ]
        )
        scenario_del_btn.click(delete_scenario, inputs=[new_scenario_box], outputs=[scenarios_dropdown])

        load_scenario_btn.click(
            lambda scenario: (
                self.scenarios[scenario]["text"], scenario,
                self.scenarios[scenario].get("base_prompt", ""),
                self.scenarios[scenario].get("base_neg_prompt", "")
            ),
            inputs=[scenarios_dropdown],
            outputs=[prompt_txt, new_scenario_box, base_prompt_txt, base_neg_prompt_txt]
        )

        copy_base_btn.click(
            copy_base_prompts_to_components, inputs=[base_prompt_txt, base_neg_prompt_txt], outputs=[base_prompt_comp, base_neg_prompt_comp]
        )
    
        return [
            checkbox_iterate, checkbox_iterate_batch, prompt_position,
            prompt_txt,
            make_combined,
            every_line_generates,
            scenarios_dropdown,
            new_scenario_box
        ]

    def run(self, p, checkbox_iterate, checkbox_iterate_batch, prompt_position, prompt_txt: str, make_combined, every_line_generates, scenarios_dropdown, new_scenario_box):
        # save prompt text to _last_scenario.json, just in case
        with open("_last_scenario.json", "w") as f:
            f.write(json.dumps({
                "text": prompt_txt,
                "prompt": p.prompt,
                "negative_prompt": p.negative_prompt
            }, indent=4))

        # generate tags for this run
        run_uuid = str(uuid.uuid4())
        scenario_name = None
        if new_scenario_box and str(new_scenario_box).strip():
            scenario_name = str(new_scenario_box).strip()
        elif scenarios_dropdown and str(scenarios_dropdown).strip():
            scenario_name = str(scenarios_dropdown).strip()
        else:
            scenario_name = "(unsaved)"

        p.extra_generation_params = p.extra_generation_params or {}
        p.extra_generation_params["scenario"] = scenario_name
        p.extra_generation_params["run_uuid"] = run_uuid

        # convert text to tree
        base = text_to_tree(prompt_txt, nest_delim='#')

        # resolve variables in the root prompt and root neg prompt
        vars_dict = {}
        neg_vars_dict = {}
        
        if p.prompt:
            _, vars_dict = replace_vars(p.prompt, vars_dict=vars_dict)
        if p.negative_prompt:
            _, neg_vars_dict = replace_vars(p.negative_prompt, vars_dict=neg_vars_dict)

        # convert tree to individual prompt lines plus var context dict per line
        lines_dicts = collect_lines(base, vars_dict=deepcopy(vars_dict), every_line_generates=every_line_generates)

        print("Got the following: ")
        for idx, (line, _) in enumerate(lines_dicts):
            print(f"{idx}: {line}")
        print()

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for line, var_dict in lines_dicts:
            if "--" in line:
                try:
                    args, rest = cmdargs(line)
                    print(f"Parsed line as commandline args: {args}, rest: '{rest}'")
                    args = {"prompt": rest, **args}
                except Exception:
                    errors.report(f"Error parsing line {line} as commandline", exc_info=True)
                    args = {"prompt": line}
            else:
                args = {"prompt": line}

            # add in the line's variables for use later
            args["var_dict"] = var_dict

            job_count += args.get("n_iter", p.n_iter)

            jobs.append(args)

        print(f"Will process {len(lines_dicts)} lines in {job_count} jobs.")
        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        for args in jobs:
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            # if p.rotate, swap width and height
            if args.get("rotate", False):
                args["width"], args["height"] = args.get("height", p.height), args.get("width", p.width)

            copy_p = copy.copy(p)
            for k, v in args.items():
                if k == "sd_model":
                    copy_p.override_settings['sd_model_checkpoint'] = v
                else:
                    setattr(copy_p, k, v)

            if args.get("prompt") and p.prompt:
                if prompt_position == "start":
                    result = args.get("prompt") + " " + p.prompt
                else:
                    result = p.prompt + " " + args.get("prompt")

                # use this line's resolved variables to peform a replacement
                # on the entire string, including the base prompt. this allows
                # us to replace variables that were initially assigned in the base.
                resolved_prompt, _ = replace_vars(result, vars_dict=deepcopy(args.get("var_dict", {})), force_retrieve=True)

                copy_p.prompt = resolved_prompt

            if args.get("negative_prompt") and p.negative_prompt:
                if prompt_position == "start":
                    copy_p.negative_prompt = args.get("negative_prompt") + " " + p.negative_prompt
                else:
                    copy_p.negative_prompt = p.negative_prompt + " " + args.get("negative_prompt")

            # allow a special symbol to disable the root prompt
            if args.get("prompt", "").startswith("?"):
                copy_p.prompt = args.get("prompt")[1:]

            proc = process_images(copy_p)

            images += proc.images

            if checkbox_iterate:
                p.seed = p.seed + (p.batch_size * p.n_iter)
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts
            # infotexts += updated_infotexts

        if make_combined and len(images) > 1:
            combined_image = image_grid(images, batch_size=1, rows=None).convert("RGB")
            full_infotext = "\n".join(infotexts)

            is_img2img = getattr(p, "init_images", None) is not None

            if opts.grid_save:  #   use grid specific Settings
                save_image(
                    combined_image,
                    opts.outdir_grids or (opts.outdir_img2img_grids if is_img2img else opts.outdir_txt2img_grids),
                    "",
                    -1,
                    prompt_txt,
                    opts.grid_format,
                    full_infotext,
                    grid=True
                )
            else:               #   use normal output Settings
                save_image(
                    combined_image,
                    opts.outdir_samples or (opts.outdir_img2img_samples if is_img2img else opts.outdir_txt2img_samples),
                    "",
                    -1,
                    prompt_txt,
                    opts.samples_format,
                    full_infotext
                )

            images.insert(0, combined_image)
            all_prompts.insert(0, prompt_txt)
            infotexts.insert(0, full_infotext)

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
