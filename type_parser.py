import copy

from nltk import pos_tag
from nltk import download
import re

# download('averaged_perceptron_tagger')


class Pos2type:
    # Takes a part-of-speech tag and converts it into a few possible initial semantic types
    def __init__(self):
        self.tag_dict = {
            'NNP': ['e'],
            'VBZ': ['N/A'],
            'VBD': ['<e,vt>', '<e,t>'],
            'DT': ['N/A'],
            'NN': ['<e,t>'],
            'JJ': ['<e,t>'],
            'RB': ['<et,et>'],
            'CC': ['N/A'],
            'CLOSE': ['<vt,t>'],
            'TO': ['<e,vt>'],
            'IN': ['<e,vt>']
        }

    def pos2type(self, sent):
        # takes in a sentence and returns a list of types
        wordlist = sent.split(' ')

        special_tags = []
        for w in wordlist:
            if w in self.tag_dict:
                special_tags.append((self.tag_dict[w], wordlist.index(w)))
                wordlist.remove(w)

        word_pos = [t[1] for t in pos_tag(wordlist)]
        pos_predicts = [self.tag_dict[t] for t in word_pos]
        for tag in reversed(special_tags):
            pos_predicts = self.insert_tag(tag, pos_predicts)

        return pos_predicts

    @staticmethod
    def insert_tag(t, predicts):
        index = t[1]
        tag = t[0]

        return predicts[0:index] + [tag] + predicts[index:]


class TypeParser:
    # A CKY parser for semantic types
    def __init__(self):
        self.grammar = self.create_grammar()
        self.p2t = Pos2type()

    def create_grammar(self):
        grammar_table = [re.sub('\n', '', line).split('\t') for line in open('type_grammar.tsv', 'r').readlines()]
        top_labels = grammar_table.pop(0)[1:]
        side_labels = [line.pop(0) for line in grammar_table]

        grammar_dict = {}
        assert len(top_labels) == len(side_labels)
        for i in range(len(side_labels)):
            for j in range(len(top_labels)):
                if not grammar_table[i][j]:
                    continue
                grammar_dict[(side_labels[i], top_labels[j])] = grammar_table[i][j]
        return grammar_dict

    def parse(self, sent):
        # takes in a sentence and creates a parse
        type_list = self.p2t.pos2type(sent)

        assert sum([len(i) for i in type_list]) == len(type_list)

        # create a parse table
        type_list = [i[0] for i in type_list]
        parse_table = [['' for i in type_list] for i in range(len(type_list))]

        # fill the bottom-level types in the diagonal of the table
        for d in range(len(type_list)):
            parse_table[d][d] = type_list[d]

        # go through additional diagonals
        for k in range(1, len(parse_table)):
            y_vals = [i for i in reversed(range(len(parse_table) - k))]
            x_vals = [i for i in reversed(range(k, len(parse_table[0])))]
            for i in range(len(x_vals)):
                # find the first daughter from the row below
                below_square = ''
                shift = 1
                while not below_square:
                    below_square = parse_table[y_vals[i] + shift][x_vals[i]]
                    b_y = y_vals[i] + shift
                    b_x = x_vals[i]
                    shift += 1

                # find a populated square on the same row to be the second daughter
                infront_square = ''
                shift = 1
                while not infront_square:
                    infront_square = parse_table[y_vals[i]][x_vals[i] - shift]
                    i_y = y_vals[i]
                    i_x = x_vals[i] - shift
                    shift += 1

                # see if the first daughter already has a parent
                try:
                    prev_square = parse_table[b_y][b_x + 1]
                except IndexError:
                    prev_square = ''
                try:
                    next_square = parse_table[i_y - 1][i_x]
                except IndexError:
                    next_square = ''

                # if the two daughters are available and in the grammar, write the parent in the current square
                if (infront_square, below_square) in self.grammar and not prev_square and not next_square:
                    newsquare = self.grammar[(infront_square, below_square)]
                    if newsquare != 't':
                        parse_table[y_vals[i]][x_vals[i]] = newsquare
                    elif y_vals[i] == 0 and x_vals[i] == len(parse_table)-1:
                        parse_table[y_vals[i]][x_vals[i]] = newsquare

        if parse_table[0][len(parse_table)-1] == 't':
            return parse_table
        else:
            return -1


class TreeConstructor:
    def __init__(self, parse_table, grammar):
        self.stack = []
        self.grammar = grammar

        self.parse_table = copy.deepcopy(parse_table)
        self.left_tree = self.parse(True)

        self.parse_table = copy.deepcopy(parse_table)
        self.right_tree = self.parse(False)

    def parse(self, left_first):
        top_node = TreeNode(0, len(self.parse_table[0])-1, self.parse_table[0][-1], None)
        self.stack.append(top_node)

        while self.stack:
            curr_node = self.node_pop()
            daughters = self.get_daughters(curr_node)

            try:
                grammar_tuple = (daughters[0].label, daughters[1].label)
            except AttributeError:
                grammar_tuple = None

            if grammar_tuple and grammar_tuple in self.grammar:
                if daughters[1]:
                    curr_node.right = daughters[1]
                if daughters[0]:
                    curr_node.left = daughters[0]

                if left_first:
                    if daughters[1]:
                        self.stack.append(daughters[1])
                    if daughters[0]:
                        self.stack.append(daughters[0])
                else:
                    if daughters[0]:
                        self.stack.append(daughters[0])
                    if daughters[1]:
                        self.stack.append(daughters[1])

        if self.get_tree_completeness(top_node) < len(self.parse_table[0]):
            return -1
        else:
            return top_node

    def get_daughters(self, node):
        right_node_label = ''
        rn_y = node.y
        rn_x = node.x
        shift = 1
        while not right_node_label:
            rn_y = node.y + shift
            try:
                right_node_label = self.parse_table[rn_y][rn_x]
            except IndexError:
                right_node_label = None
                break
            shift += 1
        if right_node_label:
            right_node = TreeNode(rn_y, rn_x, right_node_label, node)
        else:
            right_node = None

        left_node_label = ''
        ln_y = node.y
        ln_x = 0
        shift = 1
        while not left_node_label:
            ln_x = node.x - shift
            try:
                left_node_label = self.parse_table[ln_y][ln_x]
            except IndexError:
                left_node_label = None
                break
            shift += 1
        if left_node_label:
            left_node = TreeNode(ln_y, ln_x, left_node_label, node)
        else:
            left_node = None

        return [left_node, right_node]

    def node_pop(self):
        node = self.stack.pop()
        self.parse_table[node.y][node.x] = ''
        return node

    def get_tree_completeness(self, node):
        if not node.left and not node.right:
            return 1

        left_endpts = self.get_tree_completeness(node.left) if node.left else 0
        right_endpts = self.get_tree_completeness(node.right) if node.right else 0

        return left_endpts + right_endpts


class TreeNode:
    def __init__(self, y, x, label, parent):
        self.y = y
        self.x = x
        self.label = label
        self.parent = parent

        self.left = None
        self.right = None

    def __str__(self):
        return self.label


parser = TypeParser()

sentence = 'CLOSE Floyd ran for President'

print(pos_tag(sentence.split(' ')))
print(parser.p2t.pos2type(sentence))
aparse = parser.parse(sentence)
trees = TreeConstructor(aparse, parser.grammar)
print('complete')