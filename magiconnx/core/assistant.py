from itertools import chain


class Assistant():
    name2node = {}
    prev2node = {}
    next2node = {}
    gen_name = set()
    output_names = set()

    @classmethod
    def update_maps(cls, nodes=[], inputs=[], outputs=[], inits=[]):
        cls.name2node.update({node.name: node
                              for node in chain(inits, inputs, nodes, outputs)})
        cls.gen_name.update({gen.name for gen in chain(inits, inputs)})
        cls.output_names.update({out.name for out in outputs})

        for node in nodes:
            for name in node.inputs:
                cls.prev2node.setdefault(name, []).append(node)
                if name in cls.gen_name:
                    cls.name2node[name].outputs.append(node.name)
            for name in node.outputs:
                cls.next2node.setdefault(name, []).append(node)
                if name in cls.output_names:
                    cls.name2node[name].inputs.append(node.name)
        for node in chain(inits, inputs):
            for name in node.outputs:
                cls.next2node.setdefault(name, []).append(node)
        for node in outputs:
            for name in node.inputs:
                cls.prev2node.setdefault(name, []).append(node)

    @classmethod
    def remove_node(cls, node):
        pass
