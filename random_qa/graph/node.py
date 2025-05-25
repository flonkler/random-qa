import neo4j
from pydantic import BaseModel, Field
from enum import Enum
from typing import ClassVar, Literal
from typing_inspect import is_literal_type, get_args

from random_qa.utils import sanitize_multiline_string, CRS_ETRS89_UTM33N

class GraphNode(BaseModel):
    """Base class for a node in the knowledge graph"""
    id: str
    var_name: ClassVar[str] = ""
   
    def write_to_db(self, tx: neo4j.Transaction) -> neo4j.Result:
        """Creates a new labeled node with properties in the database. If a node with the same label and the same ID
        already exists, the properties will be overwritten instead. This ensures that there are no duplicates.
        """
        query = f"""
        MERGE (n:{self.__class__.__name__} {{id: $id}})
        SET n = $properties
        """
        return tx.run(query, id=self.id, properties=self.model_dump(mode="json"))
    
    @classmethod
    def get_label(cls) -> str:
        return cls.__name__

    @classmethod
    def get_properties(cls) -> tuple[str, ...]:
        return tuple(cls.model_fields.keys())

    @classmethod
    def describe(cls) -> str:
        """Generate summary of the node's metadata (i.e., label, properties and description)."""
        # Name of variable is the first letter of the node's label by default, can be overwritten to avoid duplicates
        var_name = cls.get_label().lower()[0] if cls.var_name == "" else cls.var_name
        # Use docstring of class as description for the node, format text in a more compact form
        description_default = "No description"
        description = sanitize_multiline_string(cls.__doc__ or description_default, fold_newlines=True)
        # List the node's properties, their type and description
        properties = ""
        for property, field_info in cls.model_fields.items():
            if property == "id" or field_info.exclude:
                # Exclude ID property and fields where the exclude flag is set
                continue

            field_metadata = []
            field_description = sanitize_multiline_string(
                field_info.description or description_default,
                fold_newlines=True
            )
            if is_literal_type(field_info.annotation):
                values = get_args(field_info.annotation)
                field_metadata.append(f"type: {type(values[0]).__name__}")
                field_metadata.append(f"values: " + ", ".join(repr(value) for value in values))
            elif issubclass(field_info.annotation, Enum):
                # If type is an enumeration, get the enumeration type and list the possible values
                field_metadata.append(f"type: {field_info.annotation.__base__.__name__}")
                field_metadata.append("values: " + ", ".join(repr(entry.value) for entry in field_info.annotation))
                # Use docstring as the description
                field_description = sanitize_multiline_string(
                    field_info.annotation.__doc__ or description_default,
                    fold_newlines=True
                )
            else:
                # Otherwise just use the type name (e.g., "str", "float", etc.)
                field_metadata.append(f"type: {field_info.annotation.__name__}")

            field_metadata = "; ".join(field_metadata)
            
            # Format metadata of property such that it resembles Cypher syntax
            properties += f"Node property `{var_name}.{property}` ({field_metadata}): {field_description}\n"
        
        return f"Node `({var_name}:{cls.get_label()})`: {description}\n{properties}"


class BandEnum(str, Enum):
    """Standardized frequency bands used in Germany."""
    GSM900 = "GSM900"
    B1 = "B1"
    B3 = "B3"
    B7 = "B7"
    B8 = "B8"
    B20 = "B20"
    B32 = "B32"
    N1 = "N1"
    N28 = "N28"
    N78 = "N78"

class RadioAccessTechnologyEnum(str, Enum):
    """Radio access technology (RAT) is a set of protocols, mechanisms and standards that enable wireless communication
    between mobile devices and radio communication networks."""
    GSM = "2G"
    LTE = "4G"
    NR = "5G"

class OperatorEnum(str, Enum):
    """Name of the companies that operate cellular networks in Germany."""
    VODAFONE = "Vodafone"
    TELEFONICA = "Telefonica"
    TELEKOM = "Telekom"
    EINS_UND_EINS = "1&1"

class ConstructionEnum(str, Enum):
    """Construction category that indicates whether antennas are mounted on a free-standing mast, outside of a building
    (e.g., on a roof), or inside of a building (e.g., in a large hall or stadium).
    """
    BUILDING = "Building"
    FREESTANDING = "Freestanding"
    INDOOR = "Indoor"


class ConnectionEnum(str, Enum):
    """Type of link used to connect the site with the core network. Either a fiber cable connection is used or a
    wireless connection is provided by another site via microwave antennas."""
    FIBER = "Fiber"
    MICROWAVE = "Microwave"

class ManufacturerEnum(str, Enum):
    """Companies that produce antennas"""
    HUAWEI = "Huawei"
    COMMSCOPE = "Commscope"
    KATHREIN = "Kathrein"
    ROSENBERGER = "Rosenberger"
    AMPHENOL = "Amphenol"

class Tile(GraphNode):
    """Geographical area with a fixed size of 100x100 meters arranged in a square grid. Tiles are used as a universal
    spatial reference point."""
    name: str = Field(description="Name of the tile based on its coordinates")
    area: float = Field(description="Area covered by the tile in square kilometers")

class Region(GraphNode):
    """Administrative region of city, state or country"""
    name: str = Field(description="Name of the region")
    area: float = Field(description="Area of the region in square kilometers")
    population_count: int = Field(description="Number of inhabitants living in the region")
    population_density: float = Field(description="Ratio of population count and area")

class Site(GraphNode):
    """Physical location where antennas are mounted on a tower, on a roof or inside of a building."""
    name: str = Field(
        description=(
            "Name of the site used as an identifier. "
            "Always starts with 1 followed by four digits (e.g., 10813)."
        )
    )
    coordinates: str = Field(description="GPS coordinates in the format latitude,longitude.")
    construction: ConstructionEnum
    connection: ConnectionEnum

class Operator(GraphNode):
    """Mobile network operator (MNO), sometimes referred to as provider or operator, is a company that operates and
    maintains a radio access network to provide mobile connectivity to users."""
    name: OperatorEnum

class AntennaBase(GraphNode):
    orientation: float = Field(
        description=(
            "Direction in which the antenna is oriented in degrees. The values correspond to the cardinal directions "
            "(i.e., 0° = North, 90° = East, 180° = South, 270° = West). The different antenna orientations at a site "
            "are also referred to as sectors."
        )
    )

class MobileAntenna(AntennaBase):
    """Antenna that sends and receives radio signals to provide communication services for mobile devices."""
    var_name: ClassVar[str] = "a"
    name: str = Field(
        description=(
            "Name of the antenna used as an identifier. "
            "Always starts with 'ANT' followed by six digits (e.g., ANT123456)."
        )
    )

class MicrowaveAntenna(AntennaBase):
    """Antenna that sends and receives directed radio signals for a fixed point-to-point connection."""
    var_name: ClassVar[str] = "w"
    name: str = Field(
        description=(
            "Name of the antenna used as an identifier. "
            "Always starts with 'RIFU' followed by six digits (e.g., RIFU123456)."
        )
    )

class Service(GraphNode):
    """Mobile communication service"""
    var_name: ClassVar[str] = "x"
    band: BandEnum
    rat: RadioAccessTechnologyEnum
    frequency: int = Field(description="Radio frequency in MHz.")

class Cell(GraphNode):
    """Coverage area of a single antenna that provides a mobile communication service."""
    name: str = Field(description="Name of the cell used as an identifier.")
    user_count: float = Field(description="Estimated number of users which are served by the cell.")
    area: float = Field(description="Total coverage area of the cell.")

class Manufacturer(GraphNode):
    """Company that manufactures products such as antennas, routers or mobile devices."""
    name: ManufacturerEnum
    country: str = Field(description="The country in which the company is primarily located.")

class POI(GraphNode):
    """Point of interest (POI) is a location or area with high importance such as main stations, tourist attractions,
    stadiums or other public places with high user counts."""
    name: str = Field(description="Name of the point of interest")
