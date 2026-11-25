import unittest
from pathlib import Path

import torch

from GraphSAGE.SignalAttentionData import (
    SIGNAL_NODE_FEATURE_SCHEMA,
    SIGNAL_ONLY_EDGE_FEATURE_SCHEMA,
    VPATH_SIGNAL_EDGE_FEATURE_SCHEMA,
    signalAttentionBranchesToTensor,
    vpathAttentionBranchesToTensor,
)
from GraphSAGE.graph_data import (
    NETLIST_NODE_FEATURE_SCHEMA,
    ckt_graph_to_tensor,
    extractNetlistManager,
    netlist_to_graph_tensor,
)
from GraphSAGE.PathExtraction import CircuitGraph, PortType, VertexType


SAMPLE = Path("_data/topo_NMCCB/core_05.cir")


class AttentionGraphDataTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m_manager = extractNetlistManager(SAMPLE)
        cls.m_base = ckt_graph_to_tensor(cls.m_manager.getGraph())

    def testNetlistDropsDeadBlockAndKeyColumns(self):
        graph = netlist_to_graph_tensor(SAMPLE)
        self.assertEqual(graph.node_feature_schema, NETLIST_NODE_FEATURE_SCHEMA)
        self.assertEqual(graph.x.shape[1], 10)
        self.assertNotIn("Block", graph.node_feature_schema)
        self.assertNotIn("is_key_node", graph.node_feature_schema)
        self.assertEqual(graph.node_feature_schema, SIGNAL_NODE_FEATURE_SCHEMA)
        self.assertEqual(graph.x.shape[0], self.m_base.x.shape[0])
        baseIndexes = [
            self.m_base.node_feature_schema.index(name)
            for name in NETLIST_NODE_FEATURE_SCHEMA
        ]
        self.assertTrue(torch.equal(graph.x, self.m_base.x[:, baseIndexes]))

    def testPathAttentionBranchesKeepPhysicalVertices(self):
        graph = signalAttentionBranchesToTensor(self.m_manager)
        self.assertEqual(graph.m_graph.x.shape[0], self.m_base.x.shape[0])
        self.assertEqual(graph.m_graph.node_feature_schema, SIGNAL_NODE_FEATURE_SCHEMA)
        self.assertEqual(graph.m_signalFeatureSchema, SIGNAL_ONLY_EDGE_FEATURE_SCHEMA)
        self.assertNotIn("is_key_node", graph.m_graph.node_feature_schema)
        self.assertGreaterEqual(int(graph.m_signalIndex.min()), 0)
        self.assertLess(int(graph.m_signalIndex.max()), graph.m_graph.x.shape[0])
        expectedVpathEdges = 2 * sum(
            len(path.nodes) * (len(path.nodes) - 1) // 2
            for path in self.m_manager.getVpaths()
        )
        self.assertEqual(
            graph.m_signalLabels.count("VPATH_COMEMBER"), expectedVpathEdges
        )
        self.assertGreater(graph.m_signalLabels.count("HLINK"), 0)
        self.assertEqual(
            graph.m_signalAttr.shape[1], len(SIGNAL_ONLY_EDGE_FEATURE_SCHEMA)
        )
        for name in (
            "HLINK_GM",
            "HLINK_CM",
            "HLINK_RC",
            "HLINK_OL",
            "HLINK_OT",
            "HLINK_DIRECTED",
        ):
            self.assertIn(name, graph.m_signalFeatureSchema)

    def testVpathAttentionBranchesUseDedicatedSchema(self):
        graph = vpathAttentionBranchesToTensor(self.m_manager)
        self.assertEqual(graph.m_graph.x.shape[0], self.m_base.x.shape[0])
        self.assertEqual(graph.m_signalFeatureSchema, VPATH_SIGNAL_EDGE_FEATURE_SCHEMA)
        self.assertIn("VPATH_COMEMBER", graph.m_signalLabels)
        self.assertNotIn("HLINK", graph.m_signalLabels)
        self.assertEqual(
            graph.m_signalAttr.shape[1], len(VPATH_SIGNAL_EDGE_FEATURE_SCHEMA)
        )

    def testPathAttentionRejectsBlackBoxBlock(self):
        graph = CircuitGraph()
        input_net = graph.addVertex("in", VertexType.In)
        output_net = graph.addVertex("out", VertexType.Out)
        block = graph.addVertex("x1(test)", VertexType.Block)
        graph.addEdge(input_net, block, dev_port=PortType.Norm)
        graph.addEdge(output_net, block, dev_port=PortType.Norm)

        class Manager:

            def getGraph(self):
                return graph

            def getVpaths(self):
                return ()

            def getHpaths(self):
                return ()

            def getVpathLinks(self):
                return ()

        with self.assertRaisesRegex(ValueError, "x1\\(test\\):Block"):
            signalAttentionBranchesToTensor(Manager())


if __name__ == "__main__":
    unittest.main()
