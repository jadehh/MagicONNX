from itertools import chain


class Assistant():
    name2node = {}
    prev2node = {}
    next2node = {}

    @classmethod
    def update_maps(cls, nodes=[], inputs=[], outputs=[], inits=[]):
        cls.name2node = {node.name: node
                         for node in chain(inits, inputs, nodes, outputs)}

        gen_name = {node.name for node in chain(inits, inputs)}
        out_name = {node.name for node in outputs}
        for node in nodes:
            for name in node.inputs:
                cls.prev2node.setdefault(name, []).append(node)
                if name in gen_name:
                    cls.name2node[name].outputs.append(node.name)
            for name in node.outputs:
                cls.next2node.setdefault(name, []).append(node)
                if name in out_name:
                    cls.name2node[name].inputs.append(node.name)
        for node in chain(inits, inputs):
            for name in node.outputs:
                cls.next2node.setdefault(name, []).append(node)
        for node in outputs:
            for name in node.inputs:
                cls.prev2node.setdefault(name, []).append(node)
