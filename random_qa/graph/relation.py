import neo4j
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from typing import ClassVar
from typing_extensions import Self
import re
import itertools

from random_qa.graph.node import (
    GraphNode, Site, Tile, Region, Manufacturer, Service, Cell,
    Operator, MobileAntenna, MicrowaveAntenna, POI
)
from random_qa.utils import sanitize_multiline_string

class GraphRelationMetadata(BaseModel):
    rel_type: str
    valid_pairs: list[tuple[type[GraphNode], type[GraphNode]]]

    @field_validator("rel_type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        # Ensure that type of the relationship follows the naming conventions
        if re.match(r"[A-Z_]+", value) is None:
            raise ValueError("Value must only contain uppercase letters and underscores.")
        return value


class GraphRelation(BaseModel):
    """Base class for a relation in the knowledge graph"""
    metadata: ClassVar[GraphRelationMetadata]
    in_id: str
    out_id: str
   
    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        for out_label, in_label in self.get_label_pairs():
            if self.in_id.startswith(f"{in_label}:") and self.out_id.startswith(f"{out_label}:"):
                return self

        raise ValueError(
            f"Relationship from {self.out_id} to {self.in_id} is not allowed for type {self.metadata.rel_type}"
        )

    def write_to_db(self, tx_or_session: neo4j.Transaction | neo4j.Session) -> neo4j.Result:
        """Creates a new directed relationship between two nodes. The nodes are determined based on their IDs.
        If a relationship with the same type already exists, it will be overwritten to avoid duplicates."""
        in_label = self.in_id.split(":")[0]
        out_label = self.out_id.split(":")[0]
        query = f"""
        MERGE (in:{in_label} {{id: $in_id}})
        MERGE (out:{out_label} {{id: $out_id}})
        MERGE (out)-[:{self.metadata.rel_type.upper()}]->(in)
        """
        return tx_or_session.run(query, in_id=self.in_id, out_id=self.out_id)

    @classmethod
    def get_type(cls) -> str:
        return cls.metadata.rel_type
    
    @classmethod
    def get_label_pairs(cls) -> list[tuple[str, str]]:
        return list((out_cls.__name__, in_cls.__name__) for out_cls, in_cls in cls.metadata.valid_pairs)

    @classmethod
    def describe(cls) -> str:
        """Generate summary of the relation's metadata"""
        # Use docstring of class as description for the node, format text in a more compact form
        description = sanitize_multiline_string(cls.__doc__ or "No description", fold_newlines=True)
        # Construct list of possible relationships based on the allowed label pairs
        relationships = ""
        for out_label, in_label in cls.get_label_pairs():
            relationships += f"Relationship `(:{out_label})-[:{cls.get_type()}]->(:{in_label})`\n"
        
        return f"Relationship type `{cls.metadata.rel_type}`: {description}\n{relationships}"


class LocatedInRelation(GraphRelation):
    """Geographical point or area is located within a region."""
    metadata: ClassVar[GraphRelationMetadata] = GraphRelationMetadata(
        rel_type="LOCATED_IN",
        valid_pairs=[(Site, Region), (Tile, Region), (POI, Region), (POI, Tile)],
    )

class ProducedByRelation(GraphRelation):
    """Device is produced by a company."""
    metadata: ClassVar[GraphRelationMetadata] = GraphRelationMetadata(
        rel_type="PRODUCED_BY",
        valid_pairs=[(MobileAntenna, Manufacturer), (MicrowaveAntenna, Manufacturer)],
    )

class InstalledAtRelation(GraphRelation):
    """Device is physically located at a site."""
    metadata: ClassVar[GraphRelationMetadata] = GraphRelationMetadata(
        rel_type="INSTALLED_AT",
        valid_pairs=[(MobileAntenna, Site), (MicrowaveAntenna, Site)],
    )

class AvailableInRelation(GraphRelation):
    """Service is available in a cell."""
    metadata: ClassVar[GraphRelationMetadata] = GraphRelationMetadata(
        rel_type="AVAILABLE_IN",
        valid_pairs=[(Service, Cell)],
    )

class CoveredByRelation(GraphRelation):
    """Tile is within the coverage area of a cell."""
    metadata: ClassVar[GraphRelationMetadata] = GraphRelationMetadata(
        rel_type="COVERED_BY",
        valid_pairs=[(Tile, Cell)],
    )

class ServedByRelation(GraphRelation):
    """Antenna provides mobile communication for a cell."""
    metadata: ClassVar[GraphRelationMetadata] = GraphRelationMetadata(
        rel_type="SERVED_BY",
        valid_pairs=[(Cell, MobileAntenna)],
    )

class OperatedByRelation(GraphRelation):
    """Antennas and sites are operated and maintained by an operator."""
    metadata: ClassVar[GraphRelationMetadata] = GraphRelationMetadata(
        rel_type="OPERATED_BY",
        valid_pairs=[(MobileAntenna, Operator), (MicrowaveAntenna, Operator), (Site, Operator)],
    )

class NearByRelation(GraphRelation):
    """Site is located close to a point of interest"""
    metadata: ClassVar[GraphRelationMetadata] = GraphRelationMetadata(
        rel_type="NEAR_BY",
        valid_pairs=[(Site, POI)],
    )

class ConnectedWithRelation(GraphRelation):
    """Directed point-to-point connection between two microwave antennas."""
    metadata: ClassVar[GraphRelationMetadata] = GraphRelationMetadata(
        rel_type="CONNECTED_WITH",
        valid_pairs=[(MicrowaveAntenna, MicrowaveAntenna)],
    )

class ConnectionProvidedBy(GraphRelation):
    """Site is provided access to the core network by another site via a microwave link."""
    metadata: ClassVar[GraphRelationMetadata] = GraphRelationMetadata(
        rel_type="CONNECTION_PROVIDED_BY",
        valid_pairs=[(Site, Site)],
    )
