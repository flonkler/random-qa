import re
from collections import deque
from random_qa.graph.schema import GraphSchema

# Pattern to match identifiers for labels, variable names and properties
# Reference: https://neo4j.com/docs/cypher-manual/current/syntax/naming/#_naming_rules
# Rules:
#   - can contain any alphabetical characters, numbers and underscores
#   - can begin with $ to denote parameters
#   - must not begin with a number
#   - names quoted in backticks can use any characters including whitespaces, symbols and unicode code points
# Notes:
#   - flag `re.UNICODE` must be set such that \w also matches non-english characters (e.g., "ä")
IDENTIFIER_PATTERN = r"(?:\$?(?!\d)[\w_]+|`[^`]+`)"

# Pattern to match quantifiers used relationships and path patterns
# Reference: https://neo4j.com/docs/cypher-manual/current/patterns/reference/#quantifiers
# Rules:
#  - can be either + (at least one), * (any), {n} (exactly n) or a general quantifier
#  - general quantifier have the form "{lowerBound,upperBound}" where both bounds are optional
#    (valid examples are {2,3}, {1,}, {,10} or even {,})
# Notes:
#  - this is different to variable-length relationships (e.g., -[:TYPE*0..9]->)
QUANTIFIER_PATTERN = r"(?:\+|\*|\{[0-9]+\}|\{[0-9]*,[0-9]*\})"

# Pattern to match nodes used in graph/path patterns
# Reference: https://neo4j.com/docs/cypher-manual/current/patterns/reference/#node-patterns
# Rules:
#   - must begin with "(" and end with ")"
#   - can specify a variable name for the node
#   - can specify a label, label must begin a colon ":" followed by a label term
#     - label term can include reserved characters such as "%" (wildcard), "!" (term negation), "&" (term conjunction),
#       "|" (term disjunction) as well as brackets to define the order in which the terms should be evaluated
#     - examples of valid label expressions are: ":A", ":!A", ":A|B", ":%", ":!((A|B)&C)"
#   - can contain key-value expressions surrounded by "{" and "}"
#   - can contain a WHERE clause (this is rarely used)
# Notes:
#   - regex pattern can confuse function calls (e.g., `COUNT(a)`) as nodes
NODE_PATTERN = re.compile(
    r"\("  # start of node
    r"\s*(?P<variable>" + IDENTIFIER_PATTERN + r")?" # node variable
    r"\s*(?::(?P<label>" + IDENTIFIER_PATTERN + r"))?"  # label expression
    r"\s*(?P<properties>\{.+?\})?" # property key-value expression
    r"\s*\)",  # end of node
    flags=re.IGNORECASE | re.UNICODE
)
# Pattern to match relationships between nodes
# References: https://neo4j.com/docs/cypher-manual/current/patterns/reference/#relationship-patterns
# Rules:
#   - abbreviated relationships can be "<--", "--" or "-->"
#   - full relationships begin with "-[" or "<-[" and end with "]-" or "]->"
#   - between the brackets the rules are similar to the node pattern (this even applies to the type expression,
#     although relationships can only have one type)
#   - there are two ways how relationships can be quantified, either at the end or after the relationship type
#     (e.g., "-[:TYPE*1..3]->" is equivalent to "-[:TYPE]->{1,3}")
RELATIONSHIP_PATTERN = re.compile(
    r"^\s*"
    r"(?P<left_arrow><?-)"  # start of relationship
    r"\s*(?:\["
    r"\s*(?P<variable>" + IDENTIFIER_PATTERN + r")?"  # relationship variable
    r"\s*(?::(?P<type>" + IDENTIFIER_PATTERN + r"))?"  # relationship type expression
    r"\s*(?:\*([0-9]+|[0-9]*\.\.[0-9]*)?)?"  # variable-length quantifier (unused)
    r"\s*(?P<properties>\{.+?\})?"  # property key-value expression
    r"\s*\])?"
    r"(?P<right_arrow>->?)"  # end of relationship
    r"\s*" + QUANTIFIER_PATTERN + r"?"  # quantifier (unused)
    r"\s*$",
    flags=re.IGNORECASE | re.UNICODE
)
# Pattern to find occurences when properties are accessed of variables
# TODO: Avoid collisions with procedure names (e.g., `gds.shortestPath.dijkstra.stream`)
# TODO: Handle nested key-value pairs (e.g., `n.position.lon`) ?
PROPERTY_PATTERN = re.compile(
    r"(?P<variable>" + IDENTIFIER_PATTERN + r")\.(?P<property>" + IDENTIFIER_PATTERN + r")",
    flags=re.IGNORECASE | re.UNICODE
)

class QueryValidator:
    """Validate querys against a schema"""
    def __init__(self, schema: GraphSchema):
        # Create lookup table for valid labels and their properties
        self.valid_nodes = {node.get_label(): node.get_properties() for node in schema.nodes}
        # Create lookup table for valid relationship types and their corresponding label pairs
        self.valid_relations = {rel.get_type(): rel.get_label_pairs() for rel in schema.relations}
        # Initialize attributes
        self.errors: list[str] = []
        self.original_query = ""
        self.fixed_query = ""

    @property
    def has_errors(self):
        return len(self.errors) > 0

    @property
    def has_changes(self):
        return self.original_query != self.fixed_query
  
    def _check_relation_direction(self, relation_type: str, left_label: str, right_label: str) -> str | None:
        """Check which directions are valid for a given relationship type and given labels

        Parameters:
            relation_type: Name of the relationship type. If the type is an empty string, all possible relationship
                types will be checked
            left_label: Label of the node that is on the left of the relationship
            right_label: Label of the node that is on the right of the relationship
        
        Returns: `"both"` if both directions are valid. Otherwise, `"ltr"` (left to right) and `"rtl"` (right to left)
            are returned to indicate whether the left or the right label must be the incoming or outgoing node. If the
            relationship type is not supported for the given labels, the output will be `None`.
        """
        rtl, ltr = False, False
        for valid_type, valid_label_pairs in self.valid_relations.items():
            if valid_type != relation_type and relation_type != "":
                continue

            for out_label, in_label in valid_label_pairs:
                if (in_label == right_label or right_label == "") and (out_label == left_label or left_label == ""):
                    ltr = True  # (:out)-->(:in)
                if (in_label == left_label or left_label == "") and (out_label == right_label or right_label == ""):
                    rtl = True  # (:in)<--(:out)
                
                if rtl and ltr:
                    return "both"
        
        if rtl:
            return "rtl"
        elif ltr:
            return "ltr"
    
    @staticmethod
    def _extract_identifier(match: re.Match, key: str | int = 0) -> str:
        """Helper function to extract identifier names from capture groups"""
        value = match.group(key) or ""
        return value.strip("`")

    def validate(self, query: str) -> None:
        """Checks whether a Cypher query is valid according to the schema and fixes invalid parts if possible
        
        Parameters:
            query: Cypher query
        
        Returns:
            query: Fixed Cypher query
            errors: List of errors found while parsing the query
        """
        replacements: list[tuple[int, int, str]] = []
        variables: dict[str, str] = {}
        node_pair: deque[re.Match[str]] = deque(maxlen=2)
        # Reset attributes
        self.errors = []
        self.original_query = query
        self.fixed_query = query
        # Search for node patterns in the query (e.g., `(a:Antenna {id: 1234})`)
        for node_match in re.finditer(NODE_PATTERN, query):
            variable = self._extract_identifier(node_match, "variable")
            label = self._extract_identifier(node_match, "label")
            if label != "":
                if label not in self.valid_nodes:
                    self.errors.append(f"Label {label!r} does not exist (caused by: {node_match.group(0)!r})")
                elif variable != "":
                    if variable not in variables:
                        variables[variable] = label
                    elif variables[variable] != label:
                        self.errors.append(
                            f"Variable {variable!r} has already been assigned to label {label!r} "
                            f"(caused by: {node_match.group(0)!r})"
                        )
            
            node_pair.append(node_match)
            if len(node_pair) < 2:
                continue

            # Check if there is a relationship between the current and previous node
            left_node, right_node = node_pair
            relation_match = re.match(RELATIONSHIP_PATTERN, query[left_node.end():right_node.start()])
            if relation_match is None:
                continue

            # Determine current direction of the relationship ("both" indicates undirected relationships)
            direction = "both"
            if relation_match.group("left_arrow") == "<-":
                direction = "rtl"
            elif relation_match.group("right_arrow") == "->":
                direction = "ltr"

            relation_type = self._extract_identifier(relation_match, "type")
            if relation_type != "":
                # Check if the type exists in the schema
                if relation_type not in self.valid_relations:
                    self.errors.append(
                        f"Relationship type {relation_type!r} does not exist (caused by: {relation_match.group(0)!r})"
                    )
                    continue

            # Check if the direction is supported for the given node labels
            left_label = self._extract_identifier(left_node, "label")
            right_label = self._extract_identifier(right_node, "label")

            # Lookup label based on the variable reference
            if left_label == "":
                left_label = variables.get(self._extract_identifier(left_node, "variable"), "")           
            if right_label == "":
                right_label = variables.get(self._extract_identifier(right_node, "variable"), "")
        
            valid_direction = self._check_relation_direction(relation_type, left_label, right_label)
            if valid_direction is None:
                self.errors.append(
                    f"Relationship type {relation_type!r} does not exist for labels {left_label} and {right_label} "
                    f"(caused by: {relation_match.group(0)!r})"
                )
            elif direction[::-1] == valid_direction:
                # Relationship uses the opposite direction, this can be fixed automatically
                offset = left_node.end()
                replacements.append((
                    offset + relation_match.start("left_arrow"),
                    offset + relation_match.end("left_arrow"),
                    "<-" if valid_direction == "rtl" else "-"
                ))
                replacements.append((
                    offset + relation_match.start("right_arrow"),
                    offset + relation_match.end("right_arrow"),
                    "->" if valid_direction == "ltr" else "-"
                ))

        # Apply replacements if necessary
        if len(replacements) > 0:
            self.fixed_query = ""
            offset = 0
            for start, end, replacement in replacements:
                self.fixed_query += query[offset:start] + replacement
                offset = end
            self.fixed_query += query[offset:]
        
        # Search for node properties in the query (e.g., `n.prop`)
        for prop_match in re.finditer(PROPERTY_PATTERN, query):
            variable = self._extract_identifier(prop_match, "variable")
            property = self._extract_identifier(prop_match, "property")
            if variable not in variables:
                # Skip match if variable name is unknown
                continue
            # Lookup the valid property names for the label that is associated with the variable 
            node_properties = self.valid_nodes[variables[variable]]
            if property not in node_properties:
                self.errors.append(
                    f"Node with label {variables[variable]!r} has no such property {property!r} "
                    f"(caused by: {prop_match.group(0)!r})"
                )
