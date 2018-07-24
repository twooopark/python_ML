# python_ML
Studying Machine Learning using Python

## 1. 파이썬

  - 파이썬 기본 문법


## 2. 파이썬 패키지

  - Numpy
  - Pandas
  - Matplotlib 

## 3. 지도학습 1

  - 오류 
  
  - 선형회귀
  
    - 결정계수
    - MSE
    - 잔차
    - 레버리지
    - 실습 0401
    
  - 로지스틱회귀
  
    - 혼돈행렬
    - 민감도
    - 특이도
    - 정확도
    - 실습 0402
    
  - 나이브 베이즈
  
    - 베이즈 통계법
    - 실습 0403


## 4. 비지도학습 
  
  - k-means 클러스터 알고리즘
    - 원리 : 군집의 중심점을 찾는다. 중심점으로 반복적으로 이러한 동작을 수행하며 수렴한다.
      - 지정된 클러스터의 개수 N개 만큼 최초에 중심점의 위치 N개를 랜덤으로 선정한다.
      - 각 클러스터 내부에서 클러스터의 중심점(Centroid, 뮤)과 데이터들(Xi)의 거리가 최소화 되도록 반복한다.
      - 더 이상 위 과정에 결과가 변하지 않으면,(거리가 최소가 되었다면) 반복을 마친다.
    - 유클리드 거리(Euclidean distance)
      - 두 점 사이의 거리 계산
    - 장점 : 1. 직관적인 이해 2. 모수에 대한 추정이 필요 없다
    - 단점 : 1. 노이즈 영향 크다. 2. 외상치에 민감
    - 적절한 클러스터의 개수는 어떻게 구할까?
      - 각 클러스터의 중심점과 각 클러스터의 데이터들과의 거리의 합 : tss 
      - tss를 클러스터들에 있어서 각 클러스터 중심점과 데이터의 거리의 합이라고 한다면, 클러스터의 개수가 늘어날 수록, tss는 작아질 것이다. 하지만, 점점 작게 줄어들 것이다. 그리고, 클러스터의 개수가 많으면, 각 클러스터의 특성을 나타내는데 낭비가 생겨나게 된다. 그러므로 tss를 이용해, 변곡점(elbow)을 찾아 최적의 클러스터 개수를 구한다.
      
      ```python
      # 각 클러스터의 중심점과 각 클러스터의 데이터들과의 거리의 합 : tss(=total_ss_within) 
      def total_ss_within(X, centers, clusters):
          N_clusters = centers.shape[0]
          N_columns = centers.shape[1]
          N_rows = X.shape[0]
          ref_centers = np.zeros((N_rows, N_columns))
          for n in range(N_clusters):
              indices = (clusters == n)
              for j in range(N_columns):
                  ref_centers[indices,j] = centers[n,j]
          return np.sum((X-ref_centers)**2.0)
          
      n_cluster = np.array(range(2,20))
      total_ssw = np.array([])
      for n in n_cluster:
        kmeans = KMeans(n_clusters=n)
        clusters = kmeans.fit(X).labels_
        centers = kmeans.cluster_centers_ # 각 클러스터의 중심점
        total_ssw = np.append(total_ssw, total_ss_within(X,centers,clusters)) # 각 n개의 클러스터의 중심점~요소들 거리의 합
      ```
      
    - 활용 방안 : 1. 고객 분류  2. 주식 포트폴리오 분류 전략  3. 리스크 관리
    - 실습 0501
  
  - 계층적 군집화
    - 가까운 아이템끼리 순서대로 뭉쳐가는 형식 (DP의 Bottom up 방식처럼...)
  
  - DBScan
    - 밀도에 따라 군집을 만들어 간다. eps, minPts 파라미터를 사용한다.
    - 시작 점에서 eps(반지름)까지의 거리안에 minPts 개 이상의 데이터가 있는지 확인한다.
    - minPts개 이상의 데이터가 있다면, 군집이라고 판단하고, 그 중 한 점으로 옮긴다.
    - 이를 반복적으로 실행하면, 끊기지 않고 연결되는 좌표들이 군집을 형성한다.
  
  - 큰 원형 링안에 작은 원이 있는 형태의 데이터를 군집화 하기에 적당한 방법은?
    - DBscan으로 원형 링과 작은 원을 분류할 수 있다.
    - 실습 0502
  
  - 주성분 분석(PCA)
    - 목적 : 서로 상관관계가 있는 변수를 주성분(PC)으로 변환하는 것.
    - 변수의 값들의 변동성(분산)이 큰 순서대로 정렬을 할 수 있다. 
      - ex) PC1 = 0.5*몸무게 - 0.1*신장 + 0.4*연봉
      - 주성분이 어떤 의미를 갖는지는 알기 어렵다(해석이 어렵다). 
      - 하지만, 주성분끼리 독립성을 갖기 때문에, 분석의 성능을 높일 수 있다.
      
