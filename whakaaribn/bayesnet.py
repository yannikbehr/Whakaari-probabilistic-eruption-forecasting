import os
import tempfile

import ipdb
import pysmile

import pysmile_license


class BayesNet(object):
    def __init__(self, modelfile=None, model_name=None, model_desc=None):
        self.binning = {}
        self.net = pysmile.Network()
        self.write = self.net.write_file
        if modelfile is not None:
            if not os.path.isfile(modelfile):
                raise FileNotFoundError("Can't find file " + modelfile)
            self.net.read_file(modelfile)
        else:
            if model_name is not None:
                self.net.set_name(model_name)
            if model_desc is not None:
                self.net.set_description(model_desc)

    def set_binning(self, id, binning):
        self.binning[id] = binning
        # Now add the bin values to the node's annotation
        self.add_annotation(id, str(binning))

    def add_node(
        self, id, states, cpt, description=None, long_description=None, position=None
    ):
        handle = self.net.add_node(pysmile.NodeType.CPT, id)
        if description is not None:
            self.net.set_node_name(handle, description)
        if long_description is not None:
            self.net.set_node_description(handle, long_description)
        if position is not None:
            x_pos, y_pos = position
            self.net.set_node_position(handle, x_pos, y_pos, 85, 55)
        states_count = self.net.get_outcome_count(handle)
        for i in range(0, states_count):
            self.net.set_outcome_id(handle, i, states[i])
        for i in range(states_count, len(states)):
            self.net.add_outcome(handle, states[i])
        self.net.set_node_definition(handle, cpt)
        return handle

    def add_annotation(self, id, text):
        self.net.set_node_description(id, text)

    def set_cpt(self, id, cpt):
        handle = self.net.get_node(id)
        self.net.set_node_definition(handle, cpt)

    def get_cpt(self, id):
        handle = self.net.get_node(id)
        return self.net.get_node_definition(handle)

    def add_arc(self, node1, node2):
        self.net.add_arc(node1, node2)

    def fit(
        self,
        data,
        ex_nodes=[],
        seed=None,
        randomize=False,
        uniformize=False,
        eq_sample_size=None,
    ):
        ds = pysmile.learning.DataSet()
        fd, fname = tempfile.mkstemp()
        data.to_csv(fname, na_rep="*", index=False)
        ds.read_file(fname)
        matching = ds.match_network(self.net)
        em = pysmile.learning.EM()
        if seed is not None:
            em.set_seed(seed)
        em.set_randomize_parameters(randomize)
        em.set_uniformize_parameters(uniformize)
        if eq_sample_size is not None:
            em.set_eq_sample_size(eq_sample_size)
        em.learn(ds, self.net, matching, ex_nodes)
        self.net.update_beliefs()
        os.remove(fname)

    def reset(self):
        self.net.clear_all_evidence()
        self.net.update_beliefs()

    def __str__(self):
        self.net.update_beliefs()
        msgs = []
        for nhandle in self.net.get_all_nodes():
            nid = self.net.get_node_id(nhandle)
            if self.net.is_evidence(nhandle):
                msg = "{} has evidence set to: {}"
                msg = msg.format(nid, self.net.get_evidence(nhandle))
                msgs.append(msg)
            else:
                try:
                    posteriors = self.net.get_node_value(nhandle)
                except:
                    ipdb.set_trace()
                for i in range(0, len(posteriors)):
                    msg = "P({}={}) = {}"
                    msg = msg.format(
                        nid, self.net.get_outcome_id(nhandle, i), posteriors[i]
                    )
                    msgs.append(msg)
        return "\n".join(msgs)

    def set_evidence(self, node, val):
        evidence = self.binning[node].query(val)
        if evidence is not None:
            self.net.set_evidence(node, str(evidence))
        else:
            self.net.set_evidence(node)
        self.net.update_beliefs()
