import numpy


def match_two_vectors(vec1, vec2):
    vec1_matched = []
    vec2_matched = []

    def validate_value():
        return vec2_value > 0 and vec1_value > 0

    for index, vec1_value in enumerate(vec1):
        vec2_value = vec2[index]
        if validate_value():
            vec1_matched.append(vec1_value)
            vec2_matched.append(vec2_value)

    return numpy.array(vec1_matched), numpy.array(vec2_matched)


def cosine_similarity(vec1, vec2):
    vec1_matched, vec2_matched = match_two_vectors(vec1, vec2)
    dot_product = numpy.dot(vec1_matched, vec2_matched)

    def vector_length(vec):
        return numpy.sqrt(numpy.dot(vec, vec))

    vec1_length = vector_length(vec1_matched)
    vec2_length = vector_length(vec2_matched)

    if vec1_length == 0 or vec2_length == 0:
        return 0

    cosine_sim = dot_product / (vec1_length * vec2_length)

    return cosine_sim


item_base_avg_list = 0


def item_base_cosine_sim(vec1, vec2, model):
    global item_base_avg_list
    if item_base_avg_list == 0:
        filtered_users = [[x for x in u if x > 0] for u in model]
        item_base_avg_list = [numpy.mean(u) for u in filtered_users]

    vec1_adj = numpy.subtract(vec1, item_base_avg_list)
    vec2_adj = numpy.subtract(vec2, item_base_avg_list)

    vec1_matched, vec2_matched = match_two_vectors(vec1_adj, vec2_adj)

    return cosine_similarity(vec1_matched, vec2_matched)


def pearson_correlation(vec1, vec2, vec1_avg, vec2_avg):
    vec1_matched, vec2_matched = match_two_vectors(vec1, vec2)

    vec1_adj = numpy.subtract(vec1_matched, vec1_avg)
    vec2_adj = numpy.subtract(vec2_matched, vec2_avg)

    dot_product = numpy.dot(vec1_adj, vec2_adj)

    length_vec1_vec2_adj = numpy.sqrt(numpy.dot(vec1_adj, vec1_adj) * numpy.dot(vec2_adj, vec2_adj))

    if length_vec1_vec2_adj == 0:
        return 0

    return dot_product / length_vec1_vec2_adj


def predict_by_cosine_sim(training_model, user_ratings_list, user_id, user_test_list):
    predict_rating_list = []
    weight_list = [cosine_similarity(user_ratings_list, training_model_row) for training_model_row in training_model]

    for movie_id_test in user_test_list:
        weight_sum = 0
        predict_rating = 0

        for weight, user_in_model in zip(weight_list, training_model):
            rating_in_model = user_in_model[movie_id_test]

            def is_no_rating():
                return rating_in_model == 0

            if is_no_rating():
                continue

            weight_sum += weight
            predict_rating += (weight * rating_in_model)

        def have_weight():
            return weight_sum != 0

        if have_weight():
            predict_rating /= weight_sum
        else:
            predict_rating = 3

        predict_rating_list.append(int(numpy.rint(predict_rating)))

    return correct_ratings(predict_rating_list)


def predict_by_pearson_correlation(training_model, user_ratings_list, user_id, user_test_list, p = 0):

    def calculate_training_average_for_all():
        return [numpy.average([training_model_element for training_model_element in training_model_row if training_model_element > 0])
                for training_model_row in training_model]
    training_averages_list = calculate_training_average_for_all()

    def calculate_user_average():
        return numpy.average([x for x in user_ratings_list.values() if x > 0])
    user_average = calculate_user_average()

    weight_list = [pearson_correlation(user_ratings_list, training_model_row, user_average, training_average)
                   for training_model_row, training_average in zip(training_model, training_averages_list)]

    def has_case_mod():
        return p != 0
    if has_case_mod():
        weight_list = [w * numpy.abs(w) ** (p - 1) for w in weight_list]

    predict_rating_list = []

    for movie_id_test in user_test_list:
        weight_sum = 0
        predict_plus = 0

        for weight, user_in_model, user_in_model_avg in zip(weight_list, training_model, training_averages_list):
            user_in_model_rate = user_in_model[movie_id_test]
            if user_in_model_rate == 0:
                continue

            weight_sum += numpy.abs(weight)
            predict_plus += (weight * (user_in_model_rate - user_in_model_avg))

        predict_rating = user_average

        def validate_weight_sum():
            return weight_sum != 0
        if validate_weight_sum():
            predict_rating += predict_plus / weight_sum

        predict_rating_list.append(predict_rating)

    return correct_ratings(predict_rating_list)


def correct_ratings(ratings):
    for index, rating in enumerate(ratings):
        rating = int(numpy.rint(rating))

        if rating > 5:
            print(rating)
            rating = 5
        elif rating < 1:
            print(rating)
            rating = 1

        ratings[index] = rating

    return ratings


def predict_by_item_base(training_model, user_ratings_list, user_id, user_test_list):
    model_items = numpy.array(training_model).T
    user_item_list = list(user_ratings_list.keys())

    predict_rating_list = []
    for movie_id_test in user_test_list:
        weight_sum = 0
        rating = 0
        weight_list = [item_base_cosine_sim(model_items[user_item_idx], 
            model_items[movie_id_test], training_model) for user_item_idx in user_item_list]

        for weight, user_item_idx in zip(weight_list, user_item_list):
            weight_sum += numpy.abs(weight)
            rating += (weight * user_ratings_list[user_item_idx])

        def validate_weight_sum():
            return weight_sum != 0
        if validate_weight_sum():
            rating /= weight_sum
        else:
            rating = 3

        rating = int(numpy.rint(rating))
        predict_rating_list.append(rating)

    return correct_ratings(predict_rating_list)


def predict_ratings(training_model, user_id, user_ratings_list, user_test_list, predictions):
    def validate_input():
        return len(user_test_list) > 0

    if validate_input():
        ratings = []
        rating_list1 = predict_by_cosine_sim(training_model, user_ratings_list, user_id, user_test_list)
        rating_list2 = predict_by_pearson_correlation(training_model, user_ratings_list, user_id, user_test_list)
        rating_list3 = predict_by_item_base(training_model, user_ratings_list, user_id, user_test_list)
        for rating1, rating2, rating3 in zip(rating_list1, rating_list2, rating_list3):
            ratings.append((rating1 + rating2 + rating3) / 3)

        correct_ratings(ratings)
        user_id += 1
        for index, rating in enumerate(ratings):
            if rating < 1 or rating > 5:
                rating = 3
            predictions.append((user_id, user_test_list[index] + 1, rating))


def evaluate_training_model(training_model, test_file_name, output_file_name, iuf_flag = 0):
    test_data = open(test_file_name, 'r').read().strip().split('\n')
    test_data = [data.split() for data in test_data]
    test_data = [[int(e) for e in data] for data in test_data]

    processing_user_id = test_data[0][0] - 1
    user_ratings_list = {}
    user_test_list = []
    predictions = []

    def apply_iuf_to_model():
        total_user_num = len(training_model)
        for model_movie_id in range(1000):
            movie_rate_count = len([1 for training_model_row in training_model if training_model_row[model_movie_id] != 0])
            if movie_rate_count == 0:
                continue
            iuf = numpy.log(total_user_num / movie_rate_count)
            for training_model_row in training_model:
                training_model_row[model_movie_id] *= iuf

    if iuf_flag != 0:
        apply_iuf_to_model()

    for user_id, movie_id, rating in test_data:
        user_id -= 1
        movie_id -= 1

        def new_user():
            return user_id != processing_user_id

        if new_user():
            predict_ratings(training_model, processing_user_id, user_ratings_list, user_test_list, predictions)
            user_ratings_list = {}
            user_test_list = []
            processing_user_id = user_id

        if rating != 0:
            user_ratings_list[movie_id] = rating
        else:
            user_test_list.append(movie_id)

    predict_ratings(training_model, processing_user_id, user_ratings_list, user_test_list, predictions)

    output_file = open(output_file_name, 'w')
    for row in predictions:
        output_file.write(' '.join(str(number) for number in row) + '\n')


def run():
    total_movies = 1000
    total_users = 200

    training_model = [[0] * total_movies] * total_users

    #build training model
    training_data = open('train.txt', 'r')
    training_data = training_data.read().strip()
    training_data = training_data.split('\n')

    for user_index, line in enumerate(training_data):
        training_model[user_index] = [int(rating) for rating in line.split('\t')]

    print('Evaluate test5')
    evaluate_training_model(training_model, 'test5.txt', 'result5.txt')
    print('Evaluate test10')
    evaluate_training_model(training_model, 'test10.txt', 'result10.txt')
    print('Evaluate test20')
    evaluate_training_model(training_model, 'test20.txt', 'result20.txt')

    print("eval complete")

run()

