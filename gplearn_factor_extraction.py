import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr


class FactorMining:
    def __init__(self,
                 data: pd.DataFrame,
                 features: list,
                 target_column: str,
                 target_period: int = 10,
                 population_size: int = 1000,
                 generations: int = 50,
                 function_set=('add', 'sub', 'mul', 'div',
                               'sqrt', 'log', 'abs','neg','inv','max','min',
                               'sin','cos','tan','sig'),
                 test_size: float = 0.10,
                 random_state: int = 42,
                 metric: str = 'pearson'):  # 新增 metric 参数
        """
        初始化因子挖掘类
        :param data: 包含因子的 DataFrame
        :param features: 用于训练的特征列表
        :param target_column: 目标列（用于计算收益率）
        :param target_period: 计算未来收益率的期数
        :param population_size: 种群规模
        :param generations: 进化代数
        :param function_set: 可用的数学操作符集合
        :param test_size: 测试集比例
        :param random_state: 随机种子
        :param metric: 选择优化 Rank IC ('spearman') 或 IC ('pearson')
        """
        self.data = data.copy()  # 避免修改原数据
        self.features = features
        self.target_column = target_column
        self.target_period = target_period
        self.population_size = population_size
        self.generations = generations
        self.function_set = function_set
        self.test_size = test_size
        self.random_state = random_state
        self.metric = metric.lower()  # 统一转小写，避免输入错误

        # 数据预处理
        self.X_train, self.X_test, self.y_train, self.y_test = self._preprocess_data()

    def _preprocess_data(self):
        """
        数据预处理，标准化特征并计算目标收益率
        """
        # 计算目标变量：未来 target_period 天的收益率
        self.data["target"] = self.data[self.target_column].shift(-self.target_period)/self.data[self.target_column]  - 1
        self.data = self.data.fillna(0)

        # 标准化特征
        scaler = StandardScaler()
        self.data[self.features] = scaler.fit_transform(
            self.data[self.features])

        X = self.data[self.features]
        y = self.data["target"]

        # 划分训练集与测试集
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def _rank_ic(self, y_true, y_pred):
        """
        计算 Rank IC (Spearman correlation)
        :param y_true: 实际目标值
        :param y_pred: 预测值
        :return: Rank IC 值
        """
        return spearmanr(y_true, y_pred)[0]

    def _ic(self, y_true, y_pred):
        """
        计算 IC (Pearson correlation)
        :param y_true: 实际目标值
        :param y_pred: 预测值
        :return: Pearson IC 值
        """
        # print('y_true:', len(y_true), 'y_pred:', len(y_pred))
        return pearsonr(y_true, y_pred)[0]

    def _train_gplearn(self):
        """
        使用 gplearn 训练因子，寻找最佳因子表达式
        :return: gplearn 训练模型
        """
        gp = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            function_set=self.function_set,
            metric=self.metric,  # 目标优化 Rank IC 还是 IC
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.1,
            p_point_mutation=0.1,
            verbose=1,
            random_state=self.random_state,
            stopping_criteria=1
        )

        gp.fit(self.X_train, self.y_train)
        return gp

    def evaluate_factor(self, gp):
        """
        评估因子模型的表现，并解析表达式中的特征名
        """
        y_pred = gp.predict(self.X_train)
        rank_ic = self._rank_ic(self.y_train, y_pred)
        ic = self._ic(self.y_train, y_pred)
        monotonic=self._monotonic(self.y_train,y_pred)

        # 获取原始因子表达式
        raw_expression = str(gp._program)

        # 替换 Xn 为真实特征名 (修复部分匹配问题)
        import re
        feature_map = {f'X{i}': name for i, name in enumerate(self.features)}

        # 按照数字大小倒序排列，避免短的X号影响长的X号
        sorted_keys = sorted(feature_map.keys(),
                             key=lambda x: int(x[1:]), reverse=True)

        for key in sorted_keys:
            value = feature_map[key]
            # 使用正则表达式确保完整单词匹配，避免 X3 影响 X30
            pattern = r'\b' + re.escape(key) + r'\b'
            raw_expression = re.sub(pattern, value, raw_expression)

        # print(f"Rank IC (Spearman): {rank_ic:.4f}")
        # print(f"IC (Pearson): {ic:.4f}")
        # print(f"Best Factor Expression: {raw_expression}")

        return rank_ic, ic,monotonic, raw_expression

    def evaluate_multiple_factors(self, gp, top_n=10):
        """
        评估多个因子表达式的表现
        :param gp: 训练好的 gplearn 模型
        :param top_n: 返回前 top_n 个表达式
        :return: 包含表达式、rank_ic、ic 的列表
        """
        results = []

        # 获取最终种群中的所有个体
        final_population = gp._programs[-1]  # 最后一代的种群

        # 按适应度排序（适应度越高越好）
        fitness_scores = [
            program.fitness_ for program in final_population if program is not None]
        sorted_indices = sorted(
            range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)

        # 获取前 top_n 个表达式
        for i in range(min(top_n, len(sorted_indices))):
            idx = sorted_indices[i]
            program = final_population[idx]

            if program is not None:
                # 获取表达式预测结果
                y_pred = program.execute(self.X_train.values)

                # 计算 IC 指标
                try:
                    rank_ic = self._rank_ic(self.y_train, y_pred)
                    ic = self._ic(self.y_train, y_pred)

                    # 获取原始因子表达式
                    raw_expression = str(program)

                    # 替换 Xn 为真实特征名
                    import re
                    feature_map = {f'X{i}': name for i,
                                   name in enumerate(self.features)}

                    # 按照数字大小倒序排列，避免短的X号影响长的X号
                    sorted_keys = sorted(feature_map.keys(),
                                         key=lambda x: int(x[1:]), reverse=True)

                    for key in sorted_keys:
                        value = feature_map[key]
                        # 使用正则表达式确保完整单词匹配，避免 X3 影响 X30
                        pattern = r'\b' + re.escape(key) + r'\b'
                        raw_expression = re.sub(pattern, value, raw_expression)

                    results.append({
                        'rank': i + 1,
                        'expression': raw_expression,
                        'rank_ic': rank_ic,
                        'ic': ic,
                        'fitness': program.fitness_
                    })

                except Exception as e:
                    # 如果某个表达式计算出错，跳过
                    print(f"表达式 {i+1} 计算出错: {e}")
                    continue

        return results

    def run(self):
        """运行因子挖掘流程"""
        gp = self._train_gplearn()
        rank_ic, ic, raw_expression = self.evaluate_factor(gp)
        return raw_expression, rank_ic, ic

    def run_multiple(self, top_n=10):
        """运行因子挖掘流程，返回多个表达式"""
        gp = self._train_gplearn()
        results = self.evaluate_multiple_factors(gp, top_n)
        return results
