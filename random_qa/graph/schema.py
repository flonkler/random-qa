from pydantic import BaseModel
from random_qa.graph.node import (
    GraphNode, Tile, Region, Site, Service, Manufacturer, Operator, MobileAntenna, MicrowaveAntenna, Cell, POI
)
from random_qa.graph.relation import (
    GraphRelation, LocatedInRelation, ProducedByRelation, InstalledAtRelation, AvailableInRelation, CoveredByRelation,
    ServedByRelation, OperatedByRelation, NearByRelation, ConnectedWithRelation, ConnectionProvidedBy
)

class GraphSchema(BaseModel):
    nodes: list[type[GraphNode]]
    relations: list[type[GraphRelation]]

    def describe(self) -> str:
        """Generate summary the entire schema based on its nodes and their relationships"""
        description = "\n".join(model.describe() for model in [*self.nodes, *self.relations])
        return description

kg_schema = GraphSchema(
    nodes=(Tile, Region, Site, Service, Manufacturer, Operator, MobileAntenna, MicrowaveAntenna, Cell, POI),
    relations=(
        LocatedInRelation, ProducedByRelation, InstalledAtRelation, AvailableInRelation, CoveredByRelation,
        ServedByRelation, OperatedByRelation, NearByRelation, ConnectedWithRelation, ConnectionProvidedBy
    )
)
